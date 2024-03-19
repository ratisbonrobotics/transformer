import jax
import optax
import torch
import os
import pickle
from torch.utils.data import DataLoader
from langmodel import LanguageModelJax
from tokenizer import encode_with_byte_fallback_utf8, load_vocab_from_json, VOCAB_SIZE

class TextDataset(torch.utils.data.Dataset):
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
        inputs = torch.tensor(self.dialogs[idx : idx + self.sequence_length], dtype=torch.int)
        labels = torch.tensor(self.dialogs[idx + 1: idx + self.sequence_length + 1], dtype=torch.long)
        return inputs, labels

def loss_fn(model, inputs, labels):
    logits = model(inputs)
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
    return loss

@jax.jit
def train_step(model, inputs, labels, optimizer):
    def loss_fn_with_model(model_params):
        return loss_fn(model.apply(model_params), inputs, labels)

    grad_fn = jax.value_and_grad(loss_fn_with_model)
    loss, grads = grad_fn(model.params)
    updates, optimizer_state = optimizer.update(grads, optimizer.state)
    model_params = optax.apply_updates(model.params, updates)
    return model_params, optimizer_state, loss

def train(model, dataset, optimizer, num_epochs, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_labels in dataloader:
            model.params, optimizer.state, loss = train_step(model, batch_inputs, batch_labels, optimizer)
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Usage example
vocab_size = 32768
seq_len = 2048
num_blocks = 16
num_heads = 8
hidden_dim = 768
ff_dim = 2048

model = LanguageModelJax(vocab_size, seq_len, num_blocks, num_heads, hidden_dim, ff_dim)
dataset = TextDataset("open_orca.pkl", seq_len, load_vocab_from_json("tokenizer.json"), cache_file="open_orca_cache.pkl")
optimizer = optax.adam(learning_rate=1e-4)
num_epochs = 10
batch_size = 32

train(model, dataset, optimizer, num_epochs, batch_size)