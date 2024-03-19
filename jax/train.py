import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

def loss_fn(params, batch):
    inputs, labels = batch
    logits = model(inputs)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=vocab_size)
    loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
    return loss

@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def train(model, dataset, num_epochs, batch_size, learning_rate):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            batch = jax.tree_map(lambda x: jnp.array(x), batch)
            model, opt_state, loss = train_step(model, opt_state, batch)
            epoch_loss += loss
            num_batches += 1

        epoch_loss /= num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    return model

# Hyperparameters
vocab_size = 32768
seq_len = 2048
num_blocks = 16
num_heads = 8
hidden_dim = 768
ff_dim = 2048
num_epochs = 10
batch_size = 8
learning_rate = 1e-4

# Initialize the model
model = LanguageModelJax(vocab_size, seq_len, num_blocks, num_heads, hidden_dim, ff_dim)

# Load the dataset
dataset = TextDataset("dialogs.pkl", seq_len, vocab)

# Train the model
trained_model = train(model, dataset, num_epochs, batch_size, learning_rate)