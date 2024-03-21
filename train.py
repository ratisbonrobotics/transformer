import os
import jax
import tqdm
import wandb
import pickle
import random
from model import language_model, init_params
from tokenizer import encode_with_byte_fallback_utf8, load_vocab_from_json, VOCAB_SIZE

# screen -L -S train -t train bash -c 'cd /home/markusheimerl/transformer && /bin/python3 /home/markusheimerl/transformer/train.py'

# Constants
NUM_EPOCHS = 128
SEQ_LENGTH = 2048
BATCH_SIZE = 8
WARMUP_STEPS = 1000
WANDB = True

def create_adam_state(params, learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
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
    def __init__(self, file_path, sequence_length, loaded_vocab, cache_file="dialogs_cache.pkl"):
        self.sequence_length = sequence_length
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as file:
                self.dialogs = pickle.load(file)
        else:
            with open(file_path, "rb") as file:
                loaded_dialogs = pickle.load(file)
            self.dialogs = encode_with_byte_fallback_utf8(loaded_dialogs, loaded_vocab)
            self.dialogs = [item for sublist in self.dialogs for item in sublist]
            with open(cache_file, "wb") as file:
                pickle.dump(self.dialogs, file)

    def __len__(self):
        return (len(self.dialogs) - (self.sequence_length + 1)) // self.sequence_length

    def __getitem__(self, idx):
        idx = idx * self.sequence_length
        inputs = jax.numpy.array(self.dialogs[idx : idx + self.sequence_length], dtype=jax.numpy.uint16)
        labels = jax.numpy.array(self.dialogs[idx + 1 : idx + self.sequence_length + 1], dtype=jax.numpy.uint16)
        return inputs, labels

# Create Dataset
train_dataset = TextDataset("open_orca.pkl", SEQ_LENGTH, load_vocab_from_json("tokenizer.json"), cache_file="open_orca_cache.pkl")

# Create the model
learnable_params, static_config = init_params(vocab_size=VOCAB_SIZE, seq_len=SEQ_LENGTH)
adam_state = create_adam_state(learnable_params)
if WANDB: wandb.init(project="jax")

# Define the loss function 
def loss_fn(learnable_params, inputs, labels, pos, mask, n_heads, scale):
    logits = language_model(learnable_params, inputs, pos, mask, n_heads, scale)
    one_hot_labels = jax.nn.one_hot(labels, VOCAB_SIZE)
    log_softmax_logits = jax.nn.log_softmax(logits, axis=-1)
    loss = -jax.numpy.sum(one_hot_labels * log_softmax_logits) / labels.size * 128.0
    return loss

# Define training step
def train_step(learnable_params, inputs, labels, pos, mask, n_heads, scale, adam_state):
    loss, grads = jax.value_and_grad(loss_fn)(learnable_params, inputs, labels, pos, mask, n_heads, scale)
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

jit_train_step = jax.jit(train_step, static_argnums=(5,6))

# Training loop
for epoch in range(NUM_EPOCHS):
    indices = list(range(0, len(train_dataset), BATCH_SIZE))
    random.shuffle(indices)
    with tqdm.tqdm(indices) as pbar:
        for batch_idx in pbar:
            batch_inputs, batch_labels = [], []
            for i in range(batch_idx, min(batch_idx + BATCH_SIZE, len(train_dataset))):
                inputs, labels = train_dataset[i]
                batch_inputs.append(inputs)
                batch_labels.append(labels)

            learnable_params_bfloat16 = jax.tree_util.tree_map_with_path(lambda path, param: param.astype(jax.numpy.bfloat16) if path[0] in ["tok_emb", "pos_emb", "out_linear", "in_weight", "out_weight", "q_linear", "k_linear", "v_linear", "o_linear"] else param, learnable_params)
            loss, learnable_params, adam_state = jit_train_step(learnable_params_bfloat16, jax.numpy.stack(batch_inputs), jax.numpy.stack(batch_labels), static_config['pos'], static_config['mask'], static_config['n_heads'], static_config['scale'], adam_state)
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Loss: {loss:.4f}")
            if WANDB: wandb.log({"loss": loss.item()})