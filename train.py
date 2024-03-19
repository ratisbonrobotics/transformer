import os
import tqdm
import jax
import jax.numpy as jnp
import pickle
from model import language_model, init_params
from tokenizer import encode_with_byte_fallback_utf8, load_vocab_from_json, VOCAB_SIZE

# Constants
NUM_EPOCHS = 128
SEQ_LENGTH = 256
WARMUP_STEPS = 1000
TARGET_LR = 1e-1
BATCH_SIZE = 4

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
        inputs = jnp.array(self.dialogs[idx : idx + self.sequence_length], dtype=jnp.int32)
        labels = jnp.array(self.dialogs[idx + 1 : idx + self.sequence_length + 1], dtype=jnp.int32)
        return inputs, labels

# Create Dataset
train_dataset = TextDataset("open_orca.pkl", SEQ_LENGTH, load_vocab_from_json("tokenizer.json"), cache_file="open_orca_cache.pkl")

# Create the model
learnable_params, static_config = init_params(vocab_size=VOCAB_SIZE, seq_len=SEQ_LENGTH)

# Define the loss function and 
def loss_fn(learnable_params, inputs, labels, pos, mask, n_heads):
    logits = language_model(learnable_params, inputs, pos, mask, n_heads)
    one_hot_labels = jax.nn.one_hot(labels, VOCAB_SIZE)
    log_softmax_logits = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(one_hot_labels * log_softmax_logits) / labels.size
    return loss

#jit_loss_fn = jax.jit(loss_fn, static_argnums=(3,4,5))
jit_loss_fn = loss_fn
grad_fn = jax.value_and_grad(jit_loss_fn)

# Training loop
for epoch in range(NUM_EPOCHS):
    with tqdm.tqdm(range(0, len(train_dataset), BATCH_SIZE)) as pbar:
        for batch_idx in pbar:
            batch_inputs, batch_labels = [], []
            for i in range(batch_idx, min(batch_idx + BATCH_SIZE, len(train_dataset))):
                inputs, labels = train_dataset[i]
                batch_inputs.append(inputs)
                batch_labels.append(labels)

            batch_inputs = jnp.stack(batch_inputs)
            batch_labels = jnp.stack(batch_labels)

            loss, grads = grad_fn(learnable_params, batch_inputs, batch_labels, static_config['pos'], static_config['mask'], static_config['n_heads'])
            learnable_params = jax.tree_map(lambda p, g: p - TARGET_LR * g, learnable_params, grads)

            # Log progress
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Loss: {loss:.4f}")
