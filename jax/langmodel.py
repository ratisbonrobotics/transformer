import jax
import jax.numpy as jnp
import torch
import numpy as np
from tblock import TransformerBlockTorch
from tblock import TransformerBlockJax

class LanguageModelTorch(torch.nn.Module):
    def __init__(self, vocab_size=32768, seq_len=2048, num_blocks=16, num_heads=8, hidden_dim=768, ff_dim=2048):
        super(LanguageModelTorch, self).__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = torch.nn.Embedding(seq_len, hidden_dim)
        self.register_buffer("pos", torch.arange(seq_len, dtype=torch.int))
        self.pos_norm = torch.nn.LayerNorm(hidden_dim)
        self.register_buffer("mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlockTorch(num_heads, hidden_dim, ff_dim) for _ in range(num_blocks)])
        self.out_norm = torch.nn.LayerNorm(hidden_dim)
        self.out_linear = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

        torch.nn.init.constant_(self.tok_emb.weight, 0.5)
        torch.nn.init.constant_(self.pos_emb.weight, 0.5)
        torch.nn.init.constant_(self.out_linear.weight, 0.5)

    def forward(self, token_ids: torch.Tensor):
        x = self.tok_emb(token_ids) + self.pos_emb(self.pos)
        x = self.pos_norm(x)
        
        for block in self.transformer_blocks:
            x = block(x, self.mask)

        return self.out_linear(self.out_norm(x))

class LanguageModelJax:
    def __init__(self, vocab_size=32768, seq_len=2048, num_blocks=16, num_heads=8, hidden_dim=768, ff_dim=2048):
        self.tok_emb = jnp.ones((vocab_size, hidden_dim)) * 0.5
        self.pos_emb = jnp.ones((seq_len, hidden_dim)) * 0.5
        self.pos = jnp.arange(seq_len)
        self.pos_norm_scale = jnp.ones(hidden_dim)
        self.pos_norm_bias = jnp.zeros(hidden_dim)
        self.mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1).astype(bool)
        self.transformer_blocks = [TransformerBlockJax(num_heads, hidden_dim, ff_dim) for _ in range(num_blocks)]
        self.out_norm_scale = jnp.ones(hidden_dim)
        self.out_norm_bias = jnp.zeros(hidden_dim)
        self.out_linear_weight = jnp.ones((hidden_dim, vocab_size)) * 0.5

    def __call__(self, token_ids):
        x = self.tok_emb[token_ids] + self.pos_emb[self.pos]
        x = self.layer_norm(x, self.pos_norm_scale, self.pos_norm_bias)

        for block in self.transformer_blocks:
            x = block(x, self.mask)

        x = self.layer_norm(x, self.out_norm_scale, self.out_norm_bias)
        return jnp.dot(x, self.out_linear_weight)

    def layer_norm(self, x, scale, bias):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        normalized = (x - mean) * jax.lax.rsqrt(variance + 1e-5)
        return normalized * scale + bias

def test_language_model_output():
    vocab_size = 100
    seq_len = 20
    num_blocks = 2
    num_heads = 4
    hidden_dim = 64
    ff_dim = 128
    batch_size = 2

    torch_model = LanguageModelTorch(vocab_size, seq_len, num_blocks, num_heads, hidden_dim, ff_dim)
    jax_model = LanguageModelJax(vocab_size, seq_len, num_blocks, num_heads, hidden_dim, ff_dim)

    token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    torch_input = torch.from_numpy(token_ids)
    jax_input = jnp.array(token_ids)

    torch_output = torch_model(torch_input).detach().numpy()
    jax_output = jax_model(jax_input)

    assert np.allclose(torch_output, jax_output, rtol=1e-4, atol=1e-4)

def test_language_model_output_multiple_inputs():
    vocab_size = 100
    seq_len = 20
    num_blocks = 2
    num_heads = 4
    hidden_dim = 64
    ff_dim = 128
    batch_size = 2
    num_inputs = 5

    torch_model = LanguageModelTorch(vocab_size, seq_len, num_blocks, num_heads, hidden_dim, ff_dim)
    jax_model = LanguageModelJax(vocab_size, seq_len, num_blocks, num_heads, hidden_dim, ff_dim)

    for _ in range(num_inputs):
        token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

        torch_input = torch.from_numpy(token_ids)
        jax_input = jnp.array(token_ids)

        torch_output = torch_model(torch_input).detach().numpy()
        jax_output = jax_model(jax_input)

        assert np.allclose(torch_output, jax_output, rtol=1e-4, atol=1e-4)

# Run the tests
test_language_model_output()
test_language_model_output_multiple_inputs()
print("All tests passed!")