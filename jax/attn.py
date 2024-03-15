import jax
import torch
import jax.numpy as jnp
import numpy as np

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

class AttentionJax:
    def __init__(self, n_heads, hidden_dim, head_dim):
        self.n_heads = n_heads
        self.scale = head_dim ** -0.5
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.q_linear = jnp.full((hidden_dim, n_heads * head_dim), 0.5)
        self.k_linear = jnp.full((hidden_dim, n_heads * head_dim), 0.5)
        self.v_linear = jnp.full((hidden_dim, n_heads * head_dim), 0.5)
        self.o_linear = jnp.full((n_heads * head_dim, hidden_dim), 0.5)
       
    def __call__(self, x, mask):
        batch_size, seq_len, _ = x.shape
        # Linear transformations
        q = jnp.dot(x, self.q_linear).reshape(batch_size, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = jnp.dot(x, self.k_linear).reshape(batch_size, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        v = jnp.dot(x, self.v_linear).reshape(batch_size, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        # Compute attention scores
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        scores = jnp.where(mask[:seq_len, :seq_len], -jnp.inf, scores)
        scores = jax.nn.softmax(scores, axis=-1)
        # Compute output
        output = jnp.matmul(scores, v)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = jnp.dot(output, self.o_linear)
        return output


def test_attention_output():
    # Define the attention parameters
    n_heads = 4
    hidden_dim = 64
    head_dim = 16
    seq_len = 10
    batch_size = 2

    # Create instances of both Attention modules
    torch_attn = AttentionTorch(n_heads, hidden_dim, head_dim)
    jax_attn = AttentionJax(n_heads, hidden_dim, head_dim)

    # Generate random input data and mask
    input_data = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    mask = np.random.randint(0, 2, size=(seq_len, seq_len)).astype(bool)

    # Convert input data and mask to PyTorch tensor and JAX array
    torch_input = torch.from_numpy(input_data)
    torch_mask = torch.from_numpy(mask)
    jax_input = jnp.array(input_data)
    jax_mask = jnp.array(mask)

    # Compute the output using both modules
    torch_output = torch_attn(torch_input, torch_mask).detach().numpy()
    jax_output = jax_attn(jax_input, jax_mask)

    # Check if the outputs are equal within a small tolerance
    assert np.allclose(torch_output, jax_output, rtol=1e-4, atol=1e-4)

def test_attention_output_multiple_inputs():
    # Define the attention parameters
    n_heads = 4
    hidden_dim = 64
    head_dim = 16
    seq_len = 10
    batch_size = 2

    # Create instances of both Attention modules
    torch_attn = AttentionTorch(n_heads, hidden_dim, head_dim)
    jax_attn = AttentionJax(n_heads, hidden_dim, head_dim)

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
        torch_output = torch_attn(torch_input, torch_mask).detach().numpy()
        jax_output = jax_attn(jax_input, jax_mask)

        # Check if the outputs are equal within a small tolerance
        assert np.allclose(torch_output, jax_output, rtol=1e-4, atol=1e-4)

# Run the tests
test_attention_output()
test_attention_output_multiple_inputs()
print("All tests passed!")