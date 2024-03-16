import jax
import jax.numpy as jnp
import optax
from langmodel import LanguageModelJax

def loss_fn(params, batch):
    token_ids, labels = batch
    logits = model.apply(params, token_ids)
    one_hot_labels = jax.nn.one_hot(labels, vocab_size)
    loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
    return loss

@jax.jit
def train_step(params, opt_state, batch, optimizer):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def train(model, data, num_epochs, batch_size, learning_rate):
    optimizer = optax.adam(learning_rate)
    params = model.init(jax.random.PRNGKey(0))
    opt_state = optimizer.init(params)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            params, opt_state, loss = train_step(params, opt_state, batch, optimizer)
            epoch_loss += loss

        epoch_loss /= (len(data) // batch_size)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    return params

# Example usage
vocab_size = 100
seq_len = 20
num_blocks = 2
num_heads = 2
hidden_dim = 64
ff_dim = 128

model = LanguageModelJax(vocab_size, seq_len, num_blocks, num_heads, hidden_dim, ff_dim)

# Prepare dummy training data
data = [
    (jnp.array([1, 2, 3, 4, 5]), jnp.array([2, 3, 4, 5, 6])),
    (jnp.array([6, 7, 8, 9, 10]), jnp.array([7, 8, 9, 10, 11])),
    (jnp.array([11, 12, 13, 14, 15]), jnp.array([12, 13, 14, 15, 16])),
    (jnp.array([16, 17, 18, 19, 20]), jnp.array([17, 18, 19, 20, 21])),
]

num_epochs = 3
batch_size = 2
learning_rate = 1e-3

trained_params = train(model, data, num_epochs, batch_size, learning_rate)