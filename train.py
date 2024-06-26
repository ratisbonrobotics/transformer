import jax
import tqdm
import wandb
import random
import requests
from data import TextDataset
from model import language_model, init_params
from optim import create_rmsprop_state, apply_rmsprop_optimizer

# screen -L -S train -t train bash -c 'cd /home/markusheimerl/transformer && /bin/python3 /home/markusheimerl/transformer/train.py'

# Define constants
NUM_EPOCHS = 50
BATCH_SIZE = 4
WANDB = True

# Create dataset
train_dataset = TextDataset(["books-0002-tok.pkl", "en_simple_wiki_v0-0001-tok.pkl"], sequence_length=2048)

# Create model
random_seed = random.randint(0, 2**16-1)
learnable_params, static_config = init_params(vocab_size=train_dataset.vocab_size, seq_len=train_dataset.sequence_length, dtype=jax.numpy.bfloat16, rng_key=jax.random.key(random_seed))
print(f"Total number of trainable parameters: {sum(jax.numpy.prod(jax.numpy.array(param.shape)).item() for param in jax.tree_util.tree_leaves(learnable_params))} - PRNG seed used for parameter initialization: {random_seed}")

# Create optimizer
optimizer_state = create_rmsprop_state(learnable_params, learning_rate=7e-5)

# Replicate model parameters across devices
static_config['mask'] = jax.device_put_replicated(static_config['mask'], jax.local_devices())
learnable_params = jax.device_put_replicated(learnable_params, jax.local_devices())
optimizer_state = jax.device_put_replicated(optimizer_state, jax.local_devices())

# Define loss function 
def loss_fn(learnable_params, inputs, labels, mask, batch_size, seq_len, num_heads, hidden_dim, vocab_size):
    logits = language_model(learnable_params, inputs, mask, batch_size, seq_len, num_heads, hidden_dim)
    one_hot_labels = jax.nn.one_hot(labels, vocab_size)
    log_softmax_logits = jax.nn.log_softmax(logits, axis=2)
    loss = -jax.numpy.sum(one_hot_labels * log_softmax_logits) / (seq_len * batch_size)
    return loss

# Define and compile training step
def train_step(learnable_params, optimizer_state, inputs, labels, mask, batch_size, seq_len, num_heads, hidden_dim, vocab_size):
    loss, grads = jax.value_and_grad(loss_fn)(learnable_params, inputs, labels, mask, batch_size, seq_len, num_heads, hidden_dim, vocab_size)
    grads = jax.lax.pmean(grads, axis_name='p')
    learnable_params, optimizer_state = apply_rmsprop_optimizer(learnable_params, optimizer_state, grads)
    return learnable_params, optimizer_state, loss

jit_train_step = jax.pmap(train_step, static_broadcasted_argnums=(5,6,7,8,9), axis_name='p')

# Start training loop
if WANDB: wandb.init(project="v4-8", name=f"{requests.get('https://api.ipify.org').text}_{random_seed}")
for epoch in range(NUM_EPOCHS):
    indices = list(range(epoch, len(train_dataset), BATCH_SIZE * jax.local_device_count()))[:-1]
    random.shuffle(indices)
    with tqdm.tqdm(indices) as pbar:
        for batch_idx in pbar:
            batch_inputs, batch_labels = [], []
            for i in range(batch_idx, batch_idx + BATCH_SIZE * jax.local_device_count()):
                inputs, labels = train_dataset[i]
                batch_inputs.append(inputs)
                batch_labels.append(labels)
            
            # Split the batch across devices
            device_batch_inputs = jax.numpy.stack(batch_inputs, dtype=jax.numpy.uint32).reshape(jax.local_device_count(), BATCH_SIZE, train_dataset.sequence_length)
            device_batch_labels = jax.numpy.stack(batch_labels, dtype=jax.numpy.uint32).reshape(jax.local_device_count(), BATCH_SIZE, train_dataset.sequence_length)
            
            learnable_params, optimizer_state, loss = jit_train_step(learnable_params, optimizer_state, device_batch_inputs, device_batch_labels, static_config['mask'], BATCH_SIZE, train_dataset.sequence_length, static_config['num_heads'], static_config['hidden_dim'], train_dataset.vocab_size)
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Loss: {jax.numpy.mean(loss):.4f}")
            if WANDB: wandb.log({"loss": jax.numpy.mean(loss).item()})
    
    jax.numpy.savez(f"checkpoint_{optimizer_state['step'][0]}.npz",
        learnable_params=jax.tree_util.tree_map(lambda x: x[0], learnable_params),
        static_config_mask=jax.tree_util.tree_map(lambda x: x[0], static_config['mask']),
        static_config_num_heads=static_config['num_heads'],
        static_config_hidden_dim=static_config['hidden_dim']
    )
