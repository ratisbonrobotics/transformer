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
NUM_EPOCHS = 1000
BATCH_SIZE = 32
WARMUP_STEPS = 200
WANDB = True

def create_adam_state(params, learning_rate=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    state = {
        "step": 0,
        "learning_rate": learning_rate,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "epsilon": epsilon,
        "m": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), params),
        "v": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), params),
    }
    return state

class TextDataset:
    def __init__(self, file_path, sequence_length=2048, cache_file="text_data_cache.pkl"):
        
        tokenizer = tiktoken.Encoding(
            name="cl100k_tokenizer",
            pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
            mergeable_ranks=load_tiktoken_bpe("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
            special_tokens={"[SYSTEM]": 100257, "[USER]": 100258, "[ASSISTANT]": 100259}
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
learnable_params, static_config = init_params(vocab_size=train_dataset.vocab_size, seq_len=train_dataset.sequence_length, rng_key=jax.random.PRNGKey(42))
print(f"Total number of trainable parameters: {sum(jax.numpy.prod(jax.numpy.array(param.shape)).item() for param in jax.tree_util.tree_leaves(learnable_params))}")

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
    #l2_regularization = sum(jax.numpy.sum(jax.numpy.square(param)) for param in jax.tree_util.tree_leaves(learnable_params))
    #loss = loss + 0.01 * l2_regularization
    return loss * 128.0

# Define training step
def train_step(learnable_params, inputs, labels, pos, mask, n_heads, scale, vocab_size, adam_state):
    learnable_params_bfloat16 = jax.tree_util.tree_map(lambda p: (p.astype(jax.numpy.bfloat16)), learnable_params)
    loss, grads = jax.value_and_grad(loss_fn)(learnable_params_bfloat16, inputs, labels, pos, mask, n_heads, scale, vocab_size)
    grads = jax.tree_util.tree_map(lambda g: (g.astype(jax.numpy.float32) / 128.0), grads)

    # adam optimizer
    adam_state['step'] += 1
    adam_state['m'] = jax.tree_util.tree_map(lambda m, g: (adam_state['beta_1'] * m) + (1 - adam_state['beta_1']) * g, adam_state['m'], grads)
    adam_state['v'] = jax.tree_util.tree_map(lambda v, g: (adam_state['beta_2'] * v) + (1 - adam_state['beta_2']) * (g ** 2), adam_state['v'], grads)
    m_corr = jax.tree_util.tree_map(lambda m: m / (1 - adam_state['beta_1'] ** adam_state['step']), adam_state['m'])
    v_corr = jax.tree_util.tree_map(lambda v: v / (1 - adam_state['beta_2'] ** adam_state['step']), adam_state['v'])
    updates = jax.tree_util.tree_map(lambda m, v: jax.lax.cond(adam_state['step'] <= WARMUP_STEPS, lambda _: adam_state['learning_rate'] * (adam_state['step'] / WARMUP_STEPS), lambda _: adam_state['learning_rate'], None) * m / (jax.numpy.sqrt(v) + adam_state['epsilon']), m_corr, v_corr)
    learnable_params = jax.tree_util.tree_map(lambda p, u: p - u, learnable_params, updates)

    return loss / 128.0, learnable_params, adam_state

jit_train_step = jax.pmap(train_step, static_broadcasted_argnums=(5,6,7))

# Training loop
if WANDB: wandb.init(project="next")
for epoch in range(NUM_EPOCHS):
    indices = list(range(0, len(train_dataset), BATCH_SIZE * jax.device_count()))[:-1]
    random.shuffle(indices)
    with tqdm.tqdm(indices) as pbar:
        for batch_idx in pbar:
            batch_inputs, batch_labels = [], []
            for i in range(batch_idx, batch_idx + BATCH_SIZE * jax.device_count()):
                inputs, labels = train_dataset[i]
                batch_inputs.append(inputs)
                batch_labels.append(labels)
            
            # Split the batch across devices
            device_batch_inputs = jax.numpy.stack(batch_inputs).reshape((jax.device_count(), BATCH_SIZE) + batch_inputs[0].shape)
            device_batch_labels = jax.numpy.stack(batch_labels).reshape((jax.device_count(), BATCH_SIZE) + batch_labels[0].shape)
            
            loss, learnable_params, adam_state = jit_train_step(learnable_params, device_batch_inputs, device_batch_labels, static_config['pos'], static_config['mask'], static_config["n_heads"], static_config["scale"], train_dataset.vocab_size, adam_state)
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Loss: {jax.numpy.mean(loss):.4f}")
            if WANDB: wandb.log({"loss": jax.numpy.mean(loss).item()})
