import os
import jax
import tqdm
import wandb
import pickle
import random
import requests
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from model import language_model, init_params
from optim import create_adam_state, apply_adam_optimizer, create_rmsprop_state, apply_rmsprop_optimizer, create_sm3_state, apply_sm3_optimizer

# screen -L -S train -t train bash -c 'cd /home/markusheimerl/transformer && /bin/python3 /home/markusheimerl/transformer/train.py'

# Constants
NUM_EPOCHS = 10
BATCH_SIZE = 2
WARMUP_STEPS = 8000
WANDB = True

class TextDataset:
    def __init__(self, file_path, sequence_length=2048, cache_file="text_data_cache.pkl"):
        
        tokenizer = tiktoken.Encoding(
            name="cl100k_tokenizer",
            pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
            mergeable_ranks=load_tiktoken_bpe("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
            special_tokens={"<|system|>": 100257, "<|user|>": 100258, "<|assistant|>": 100259, "<|endoftext|>": 100260}
        )

        self.vocab_size = tokenizer.n_vocab
        self.sequence_length = sequence_length

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as file:
                self.text_data = pickle.load(file)
        else:
            with open(file_path, "rb") as file:
                loaded_text_data = pickle.load(file)

            self.text_data = tokenizer.encode_batch(loaded_text_data, num_threads=32, allowed_special="all")
            self.text_data = [item for sublist in self.text_data for item in sublist]
            with open(cache_file, "wb") as file:
                pickle.dump(self.text_data, file)

    def __len__(self):
        return (len(self.text_data) - (self.sequence_length + 1)) // self.sequence_length

    def __getitem__(self, idx):
        idx = idx * self.sequence_length
        inputs = jax.numpy.array(self.text_data[idx : idx + self.sequence_length], dtype=jax.numpy.uint32)
        labels = jax.numpy.array(self.text_data[idx + 1 : idx + self.sequence_length + 1], dtype=jax.numpy.uint32)
        return inputs, labels

# Create Dataset
train_dataset = TextDataset("open_orca.pkl", cache_file="open_orca_cache.pkl")

# Create the model
random_seed = random.randint(0, 2**16-1)
learnable_params, static_config = init_params(vocab_size=train_dataset.vocab_size, seq_len=train_dataset.sequence_length, rng_key=jax.random.key(random_seed))
print(f"Total number of trainable parameters: {sum(jax.numpy.prod(jax.numpy.array(param.shape)).item() for param in jax.tree_util.tree_leaves(learnable_params))} - PRNG seed used for parameter initialization: {random_seed}")

# Create optimizer
optimizer_state = create_rmsprop_state(learnable_params)

# Replicate model parameters across devices
static_config['pos'] = jax.device_put_replicated(static_config['pos'], jax.local_devices())
static_config['mask'] = jax.device_put_replicated(static_config['mask'], jax.local_devices())
learnable_params = jax.device_put_replicated(learnable_params, jax.local_devices())
optimizer_state = jax.device_put_replicated(optimizer_state, jax.local_devices())

# Define the loss function 
def loss_fn(learnable_params, inputs, labels, pos, mask, n_heads, scale, vocab_size):
    logits = language_model(learnable_params, inputs, pos, mask, n_heads, scale)
    one_hot_labels = jax.nn.one_hot(labels, vocab_size)
    log_softmax_logits = jax.nn.log_softmax(logits, axis=-1)
    loss = -jax.numpy.sum(one_hot_labels * log_softmax_logits) / labels.size
    # l2 loss
    loss += 4e-6 * jax.tree_util.tree_reduce(lambda x, y: x + y, jax.tree_util.tree_map(lambda p: jax.numpy.sum(p), jax.tree_util.tree_map(lambda p: jax.numpy.square(p), learnable_params)))
    return loss * 1024.0

# Define training step
def train_step(learnable_params, adam_state, inputs, labels, pos, mask, n_heads, scale, vocab_size, total_steps):
    # decrease precision
    learnable_params_bfloat16 = jax.tree_util.tree_map(lambda p: (p.astype(jax.numpy.bfloat16)), learnable_params)
    # calculate loss
    loss, grads = jax.value_and_grad(loss_fn)(learnable_params_bfloat16, inputs, labels, pos, mask, n_heads, scale, vocab_size)
    # increase precision
    grads = jax.tree_util.tree_map(lambda g: (g.astype(jax.numpy.float32) / 1024.0), grads)
    # exchange gradients
    grads = jax.lax.pmean(grads, axis_name='p')
    # optimize
    learnable_params, adam_state, learning_rate = apply_rmsprop_optimizer(learnable_params, adam_state, grads, WARMUP_STEPS, total_steps)
    # return results
    return learnable_params, adam_state, jax.lax.pmean(loss, axis_name='p') / 1024.0, learning_rate

jit_train_step = jax.pmap(train_step, static_broadcasted_argnums=(6,7,8,9), axis_name='p')

# Training loop
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
            
            learnable_params, optimizer_state, loss, learning_rate = jit_train_step(learnable_params, optimizer_state, device_batch_inputs, device_batch_labels, static_config['pos'], static_config['mask'], static_config["n_heads"], static_config["scale"], train_dataset.vocab_size, len(indices) * NUM_EPOCHS)
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Loss: {jax.numpy.mean(loss):.4f} - Learning Rate: {jax.numpy.mean(learning_rate):.10f}")
            if WANDB: wandb.log({"loss": jax.numpy.mean(loss).item(), "learning_rate": jax.numpy.mean(learning_rate).item()})
    
    jax.numpy.savez(f"checkpoint_{optimizer_state['step'][0]}.npz",
        learnable_params=jax.tree_util.tree_map(lambda x: x[0], learnable_params),
        static_config_pos=jax.tree_util.tree_map(lambda x: x[0], static_config['pos']),
        static_config_mask=jax.tree_util.tree_map(lambda x: x[0], static_config['mask']),
        static_config_n_heads=static_config["n_heads"],
        static_config_scale=static_config["scale"]
    )
