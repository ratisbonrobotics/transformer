import jax
import jax.numpy as jnp
import torch
import numpy as np
from ffwd import FeedForwardJax
from attn import AttentionJax

class FeedForwardTorch(torch.nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.in_linear = torch.nn.Linear(hidden_dim, ff_dim, bias=False)
        self.out_linear = torch.nn.Linear(ff_dim, hidden_dim, bias=False)
        torch.nn.init.constant_(self.in_linear.weight, 0.5)
        torch.nn.init.constant_(self.out_linear.weight, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_linear(torch.nn.functional.gelu(self.in_linear(x), approximate='tanh'))

class AttentionTorch(torch.nn.Module):
    def __init__(self, n_heads, hidden_dim, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.scale = head_dim**-0.5
        self.q_linear = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.k_linear = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.v_linear = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.o_linear = torch.nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

        torch.nn.init.constant_(self.q_linear.weight, 0.5)
        torch.nn.init.constant_(self.k_linear.weight, 0.5)
        torch.nn.init.constant_(self.v_linear.weight, 0.5)
        torch.nn.init.constant_(self.o_linear.weight, 0.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        scores = scores.masked_fill(mask[:seq_len, :seq_len], float('-inf'))
        scores = torch.nn.functional.softmax(scores, dim=-1)

        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_linear(output)

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