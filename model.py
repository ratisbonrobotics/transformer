import jax

class FeedForward:
    def __init__(self, hidden_dim, ff_dim, rng_key):
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        in_scale = jax.nn.initializers.he_normal()(rng_key, (hidden_dim, ff_dim))
        out_scale = jax.nn.initializers.glorot_uniform()(rng_key, (ff_dim, hidden_dim))
        self.in_weight = jax.numpy.full((hidden_dim, ff_dim), in_scale)
        self.out_weight = jax.numpy.full((ff_dim, hidden_dim), out_scale)

    def __call__(self, x):
        x = jax.numpy.dot(x, self.in_weight)
        x = jax.nn.gelu(x, approximate=True)
        x = jax.numpy.dot(x, self.out_weight)
        return x

class Attention:
    def __init__(self, n_heads, hidden_dim, head_dim, rng_key):
        self.n_heads = n_heads
        self.scale = head_dim ** -0.5
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim

        xavier_scale = jax.nn.initializers.glorot_uniform()(rng_key, (hidden_dim, n_heads * head_dim))
        self.q_linear = jax.numpy.full((hidden_dim, n_heads * head_dim), xavier_scale)
        self.k_linear = jax.numpy.full((hidden_dim, n_heads * head_dim), xavier_scale)
        self.v_linear = jax.numpy.full((hidden_dim, n_heads * head_dim), xavier_scale)
        self.o_linear = jax.numpy.full((n_heads * head_dim, hidden_dim), xavier_scale)
       
    def __call__(self, x, mask):
        batch_size, seq_len, _ = x.shape
        # Linear transformations
        q = jax.numpy.dot(x, self.q_linear).reshape(batch_size, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = jax.numpy.dot(x, self.k_linear).reshape(batch_size, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        v = jax.numpy.dot(x, self.v_linear).reshape(batch_size, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        # Compute attention scores
        scores = jax.numpy.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        scores = jax.numpy.where(mask[:seq_len, :seq_len], -jax.numpy.inf, scores)
        scores = jax.nn.softmax(scores, axis=-1)
        # Compute output
        output = jax.numpy.matmul(scores, v)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = jax.numpy.dot(output, self.o_linear)
        return output

class TransformerBlock:
    def __init__(self, num_heads, hidden_dim, ff_dim):
        self.attention = Attention(num_heads, hidden_dim, hidden_dim // num_heads)
        self.feed_forward = FeedForward(hidden_dim, ff_dim)
        self.attention_norm_scale = jax.numpy.ones(hidden_dim)
        self.attention_norm_bias = jax.numpy.zeros(hidden_dim)
        self.ffn_norm_scale = jax.numpy.ones(hidden_dim)
        self.ffn_norm_bias = jax.numpy.zeros(hidden_dim)

    def __call__(self, x, mask):
        r = self.attention(self.layer_norm(x, self.attention_norm_scale, self.attention_norm_bias), mask)
        h = x + r
        r = self.feed_forward(self.layer_norm(h, self.ffn_norm_scale, self.ffn_norm_bias))
        out = h + r
        return out

    def layer_norm(self, x, scale, bias):
        mean = jax.numpy.mean(x, axis=-1, keepdims=True)
        variance = jax.numpy.mean(jax.numpy.square(x - mean), axis=-1, keepdims=True)
        normalized = (x - mean) * jax.lax.rsqrt(variance + 1e-5)
        return normalized * scale + bias

class LanguageModel:
    def __init__(self, vocab_size=32768, seq_len=2048, num_blocks=16, num_heads=8, hidden_dim=768, ff_dim=2048, rng_key=jax.random.PRNGKey(0)):
        rng_key, subkey = jax.random.split(rng_key)
        self.tok_emb = jax.random.normal(subkey, (vocab_size, hidden_dim)) * 0.02
        self.pos_emb = jax.random.normal(subkey, (vocab_size, hidden_dim)) * 0.02
        self.pos = jax.numpy.arange(seq_len)
        self.pos_norm_scale = jax.numpy.ones(hidden_dim)
        self.pos_norm_bias = jax.numpy.zeros(hidden_dim)
        self.mask = jax.numpy.triu(jax.numpy.ones((seq_len, seq_len)), k=1).astype(bool)
        
        self.transformer_blocks = []
        for _ in range(num_blocks):
            rng_key, block_key = jax.random.split(rng_key)
            self.transformer_blocks.append(TransformerBlock(num_heads, hidden_dim, ff_dim, block_key))

        self.out_norm_scale = jax.numpy.ones(hidden_dim)
        self.out_norm_bias = jax.numpy.zeros(hidden_dim)
        self.out_linear_weight = jax.nn.initializers.glorot_uniform()(rng_key, (hidden_dim, vocab_size))

    def __call__(self, token_ids):
        x = self.tok_emb[token_ids] + self.pos_emb[self.pos]
        x = self.layer_norm(x, self.pos_norm_scale, self.pos_norm_bias)

        for block in self.transformer_blocks:
            x = block(x, self.mask)

        x = self.layer_norm(x, self.out_norm_scale, self.out_norm_bias)
        return jax.numpy.dot(x, self.out_linear_weight)

    def layer_norm(self, x, scale, bias):
        mean = jax.numpy.mean(x, axis=-1, keepdims=True)
        variance = jax.numpy.mean(jax.numpy.square(x - mean), axis=-1, keepdims=True)
        normalized = (x - mean) * jax.lax.rsqrt(variance + 1e-5)
        return normalized * scale + bias