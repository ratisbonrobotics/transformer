import jax
import jax.numpy as jnp
import torch
import numpy as np
from ffwd import FeedForwardJax, FeedForwardTorch
from attn import AttentionJax, AttentionTorch

class TransformerBlockTorch(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim, ff_dim):
        super().__init__()
        self.attention = AttentionTorch(num_heads, hidden_dim, hidden_dim // num_heads)
        self.feed_forward = FeedForwardTorch(hidden_dim, ff_dim)
        self.attention_norm = torch.nn.LayerNorm(hidden_dim)
        self.ffn_norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), mask)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out

class TransformerBlockJax:
    def __init__(self, num_heads, hidden_dim, ff_dim):
        self.attention = AttentionJax(num_heads, hidden_dim, hidden_dim // num_heads)
        self.feed_forward = FeedForwardJax(hidden_dim, ff_dim)
        self.attention_norm_scale = jnp.ones(hidden_dim)
        self.attention_norm_bias = jnp.zeros(hidden_dim)
        self.ffn_norm_scale = jnp.ones(hidden_dim)
        self.ffn_norm_bias = jnp.zeros(hidden_dim)

    def __call__(self, x, mask):
        r = self.attention(self.layer_norm(x, self.attention_norm_scale, self.attention_norm_bias), mask)
        h = x + r
        r = self.feed_forward(self.layer_norm(h, self.ffn_norm_scale, self.ffn_norm_bias))
        out = h + r
        return out

    def layer_norm(self, x, scale, bias):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        normalized = (x - mean) * jax.lax.rsqrt(variance + 1e-5)
        return normalized * scale + bias

def test_transformer_block_output():
    # Define the transformer block parameters
    num_heads = 4
    hidden_dim = 64
    ff_dim = 128
    seq_len = 10
    batch_size = 2

    # Create instances of both TransformerBlock modules
    torch_block = TransformerBlockTorch(num_heads, hidden_dim, ff_dim)
    jax_block = TransformerBlockJax(num_heads, hidden_dim, ff_dim)

    # Generate random input data and mask
    input_data = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    mask = np.random.randint(0, 2, size=(seq_len, seq_len)).astype(bool)

    # Convert input data and mask to PyTorch tensor and JAX array
    torch_input = torch.from_numpy(input_data)
    torch_mask = torch.from_numpy(mask)
    jax_input = jnp.array(input_data)
    jax_mask = jnp.array(mask)

    # Compute the output using both modules
    torch_output = torch_block(torch_input, torch_mask).detach().numpy()
    jax_output = jax_block(jax_input, jax_mask)

    # Check if the outputs are equal within a small tolerance
    assert np.allclose(torch_output, jax_output, rtol=1e-4, atol=1e-4)

def test_transformer_block_output_multiple_inputs():
    # Define the transformer block parameters
    num_heads = 4
    hidden_dim = 64
    ff_dim = 128
    seq_len = 10
    batch_size = 2

    # Create instances of both TransformerBlock modules
    torch_block = TransformerBlockTorch(num_heads, hidden_dim, ff_dim)
    jax_block = TransformerBlockJax(num_heads, hidden_dim, ff_dim)

    # Generate multiple random input data points and masks
    num_inputs = 5
    input_data = [np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32) for _ in range(num_inputs)]
    masks = [np.random.randint(0, 2, size=(seq_len, seq_len)).astype(bool) for _ in range(num_inputs)]

    for data, mask in zip(input_data, masks):
        # Convert input data and mask to PyTorch tensor and JAX array
        torch_input = torch.from_numpy(data)
        torch_mask = torch.from_numpy(mask)
        jax_input = jnp.array(data)
        jax_mask = jnp.array(mask)

        # Compute the output using both modules
        torch_output = torch_block(torch_input, torch_mask).detach().numpy()
        jax_output = jax_block(jax_input, jax_mask)

        # Check if the outputs are equal within a small tolerance
        assert np.allclose(torch_output, jax_output, rtol=1e-4, atol=1e-4)

# Run the tests
test_transformer_block_output()
test_transformer_block_output_multiple_inputs()
print("All tests passed!")