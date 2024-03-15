import jax
import jax.numpy as jnp
import torch
import numpy as np

class FeedForwardTorch(torch.nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.in_linear = torch.nn.Linear(hidden_dim, ff_dim, bias=False)
        self.out_linear = torch.nn.Linear(ff_dim, hidden_dim, bias=False)
        torch.nn.init.constant_(self.in_linear.weight, 0.5)
        torch.nn.init.constant_(self.out_linear.weight, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_linear(torch.nn.functional.gelu(self.in_linear(x), approximate='tanh'))

class FeedForwardJax:
    def __init__(self, hidden_dim, ff_dim):
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.in_weight = jnp.full((hidden_dim, ff_dim), 0.5)
        self.out_weight = jnp.full((ff_dim, hidden_dim), 0.5)

    def __call__(self, x):
        x = jnp.dot(x, self.in_weight)
        x = jax.nn.gelu(x, approximate=True)
        x = jnp.dot(x, self.out_weight)
        return x

def test_feedforward_output():
    # Define the hidden dimension and feedforward dimension
    hidden_dim = 64
    ff_dim = 128

    # Create instances of both FeedForward modules
    torch_ff = FeedForwardTorch(hidden_dim, ff_dim)
    jax_ff = FeedForwardJax(hidden_dim, ff_dim)

    # Generate random input data
    batch_size = 10
    input_data = np.random.randn(batch_size, hidden_dim).astype(np.float32)

    # Convert input data to PyTorch tensor and JAX array
    torch_input = torch.from_numpy(input_data)
    jax_input = jnp.array(input_data)

    # Compute the output using both modules
    torch_output = torch_ff(torch_input).detach().numpy()
    jax_output = jax_ff(jax_input)

    # Check if the outputs are equal within a small tolerance
    assert np.allclose(torch_output, jax_output, rtol=1e-3, atol=1e-3)

def test_feedforward_output_multiple_inputs():
    # Define the hidden dimension and feedforward dimension
    hidden_dim = 64
    ff_dim = 128

    # Create instances of both FeedForward modules
    torch_ff = FeedForwardTorch(hidden_dim, ff_dim)
    jax_ff = FeedForwardJax(hidden_dim, ff_dim)

    # Generate multiple random input data points
    num_inputs = 5
    batch_size = 10
    input_data = [np.random.randn(batch_size, hidden_dim).astype(np.float32) for _ in range(num_inputs)]

    for data in input_data:
        # Convert input data to PyTorch tensor and JAX array
        torch_input = torch.from_numpy(data)
        jax_input = jnp.array(data)

        # Compute the output using both modules
        torch_output = torch_ff(torch_input).detach().numpy()
        jax_output = jax_ff(jax_input)

        # Check if the outputs are equal within a small tolerance
        assert np.allclose(torch_output, jax_output, rtol=1e-3, atol=1e-3)

# Run the tests
test_feedforward_output()
test_feedforward_output_multiple_inputs()
print("All tests passed!")