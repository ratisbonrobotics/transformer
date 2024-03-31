import os
import jax
import tqdm
import wandb
import pickle
import random
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from model import language_model, init_params

# screen -L -S train -t train bash -c 'cd /home/markusheimerl/transformer && /bin/python3 /home/markusheimerl/transformer/train.py'

# Constants
NUM_EPOCHS = 10
BATCH_SIZE = 2
WARMUP_STEPS = 8000
WANDB = True

def create_adam_state(params, learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    return {"step": 0, "learning_rate": learning_rate, "beta_1": beta_1, "beta_2": beta_2, "epsilon": epsilon, "m": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), params), "v": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), params)}

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
adam_state = create_adam_state(learnable_params)

# Replicate model parameters across devices
static_config['pos'] = jax.device_put_replicated(static_config['pos'], jax.local_devices())
static_config['mask'] = jax.device_put_replicated(static_config['mask'], jax.local_devices())
learnable_params = jax.device_put_replicated(learnable_params, jax.local_devices())
adam_state = jax.device_put_replicated(adam_state, jax.local_devices())

# Define the loss function 
def loss_fn(learnable_params, inputs, labels, pos, mask, n_heads, scale, vocab_size):
    logits = language_model(learnable_params, inputs, pos, mask, n_heads, scale)
    one_hot_labels = jax.nn.one_hot(labels, vocab_size)
    log_softmax_logits = jax.nn.log_softmax(logits, axis=-1)
    loss = -jax.numpy.sum(one_hot_labels * log_softmax_logits) / labels.size
    # l2 loss
    loss += 5e-6 * jax.tree_util.tree_reduce(lambda x, y: x + y, jax.tree_util.tree_map(lambda p: jax.numpy.sum(p), jax.tree_util.tree_map(lambda p: jax.numpy.square(p), learnable_params)))
    return loss * 256.0

# Define training step
def train_step(learnable_params, adam_state, inputs, labels, pos, mask, n_heads, scale, vocab_size, total_steps):
    # lower precision
    learnable_params_bfloat16 = jax.tree_util.tree_map(lambda p: (p.astype(jax.numpy.bfloat16)), learnable_params)
    # calculate loss
    loss, grads = jax.value_and_grad(loss_fn)(learnable_params_bfloat16, inputs, labels, pos, mask, n_heads, scale, vocab_size)
    # gradient scaling
    grads = jax.tree_util.tree_map(lambda g: (g.astype(jax.numpy.float32) / 256.0), grads)
    # exchange gradients
    grads = jax.lax.pmean(grads, axis_name='p')
    # adam optimizer
    adam_state['step'] += 1
    adam_state['m'] = jax.tree_util.tree_map(lambda m, g: (adam_state['beta_1'] * m) + (1 - adam_state['beta_1']) * g, adam_state['m'], grads)
    adam_state['v'] = jax.tree_util.tree_map(lambda v, g: (adam_state['beta_2'] * v) + (1 - adam_state['beta_2']) * (g ** 2), adam_state['v'], grads)
    m_corr = jax.tree_util.tree_map(lambda m: m / (1 - adam_state['beta_1'] ** adam_state['step']), adam_state['m'])
    v_corr = jax.tree_util.tree_map(lambda v: v / (1 - adam_state['beta_2'] ** adam_state['step']), adam_state['v'])
    learning_rate = jax.lax.cond(adam_state['step'] <= WARMUP_STEPS, lambda _: adam_state['learning_rate'] * (adam_state['step'] / WARMUP_STEPS), lambda _: 0.5 * adam_state['learning_rate'] * (1 + jax.numpy.cos(jax.numpy.pi * (jax.numpy.minimum(adam_state['step'] / total_steps, 1.0)))), None)
    updates = jax.tree_util.tree_map(lambda m, v: learning_rate * m / (jax.numpy.sqrt(v) + adam_state['epsilon']), m_corr, v_corr)
    learnable_params = jax.tree_util.tree_map(lambda p, u: p - u, learnable_params, updates)

    return learnable_params, adam_state, jax.lax.pmean(loss, axis_name='p') / 128.0, learning_rate

jit_train_step = jax.pmap(train_step, static_broadcasted_argnums=(6,7,8,9), axis_name='p')

# Training loop
if WANDB: wandb.init(project="v2-8")
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
            
            learnable_params, adam_state, loss, learning_rate = jit_train_step(learnable_params, adam_state, device_batch_inputs, device_batch_labels, static_config['pos'], static_config['mask'], static_config["n_heads"], static_config["scale"], train_dataset.vocab_size, len(indices) * NUM_EPOCHS)
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Loss: {jax.numpy.mean(loss):.4f} - Learning Rate: {jax.numpy.mean(learning_rate):.10f}")
            if WANDB: wandb.log({"loss": jax.numpy.mean(loss).item(), "learning_rate": jax.numpy.mean(learning_rate).item()})
    
    jax.numpy.savez(f"checkpoint_{adam_state['step'][0]}.npz",
        learnable_params=jax.tree_util.tree_map(lambda x: x[0], learnable_params),
        static_config_pos=jax.tree_util.tree_map(lambda x: x[0], static_config['pos']),
        static_config_mask=jax.tree_util.tree_map(lambda x: x[0], static_config['mask']),
        static_config_n_heads=static_config["n_heads"],
        static_config_scale=static_config["scale"]
    )
