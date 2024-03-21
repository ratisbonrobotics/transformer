import jax

def feed_forward(params, x):
    x = jax.numpy.dot(x, params['in_weight'])
    x = jax.nn.gelu(x, approximate=True)
    x = jax.numpy.dot(x, params['out_weight'])
    return x

def attention(params, x, mask, n_heads, scale):
    batch_size, seq_len, _ = x.shape
    # Linear transformations
    q = jax.numpy.dot(x, params['q_linear']).reshape(batch_size, seq_len, n_heads, -1).transpose(0, 2, 1, 3)
    k = jax.numpy.dot(x, params['k_linear']).reshape(batch_size, seq_len, n_heads, -1).transpose(0, 2, 1, 3)
    v = jax.numpy.dot(x, params['v_linear']).reshape(batch_size, seq_len, n_heads, -1).transpose(0, 2, 1, 3)
    # Compute attention scores
    scores = jax.numpy.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    scores = jax.numpy.where(mask[:seq_len, :seq_len], -jax.numpy.inf, scores)
    scores = jax.nn.softmax(scores, axis=-1)
    # Compute output
    output = jax.numpy.matmul(scores, v)
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    output = jax.numpy.dot(output, params['o_linear'])
    return output

def transformer_block(params, x, mask, n_heads, scale):
    r = attention(params['attention'], simple_rms_norm(x), mask, n_heads, scale)
    h = x + r
    r = feed_forward(params['feed_forward'], simple_rms_norm(h))
    out = h + r
    return out

def simple_rms_norm(x, eps=1e-5):
    var = jax.numpy.mean(jax.numpy.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + eps)

def language_model(params, token_ids, pos, mask, n_heads, scale):
    x = params['tok_emb'][token_ids] + params['pos_emb'][pos]
    x = simple_rms_norm(x)
    for block_params in params['transformer_blocks']:
        x = transformer_block(block_params, x, mask, n_heads, scale)
    x = simple_rms_norm(x)
    return jax.numpy.dot(x, params['out_linear'])

def init_params(vocab_size, seq_len, num_blocks=16, num_heads=8, hidden_dim=768, ff_dim=2048, rng_key=jax.random.PRNGKey(0)):
    rng_key, subkey = jax.random.split(rng_key)
    xavier_uniform_init = jax.nn.initializers.glorot_uniform(dtype=jax.numpy.float32)
    kaiming_normal_init = jax.nn.initializers.he_normal(dtype=jax.numpy.float32)

    learnable_params = {
        'tok_emb': jax.random.normal(subkey, (vocab_size, hidden_dim), dtype=jax.numpy.float32) * 0.02,
        'pos_emb': jax.random.normal(subkey, (vocab_size, hidden_dim), dtype=jax.numpy.float32) * 0.02,
        'transformer_blocks': [],
        'out_linear': xavier_uniform_init(rng_key, (hidden_dim, vocab_size), dtype=jax.numpy.float32),
    }

    for _ in range(num_blocks):
        rng_key, block_key = jax.random.split(rng_key)
        block_params = {
            'attention': {
                'q_linear': xavier_uniform_init(block_key, (hidden_dim, num_heads * (hidden_dim // num_heads)), dtype=jax.numpy.float32),
                'k_linear': xavier_uniform_init(block_key, (hidden_dim, num_heads * (hidden_dim // num_heads)), dtype=jax.numpy.float32),
                'v_linear': xavier_uniform_init(block_key, (hidden_dim, num_heads * (hidden_dim // num_heads)), dtype=jax.numpy.float32),
                'o_linear': xavier_uniform_init(block_key, (num_heads * (hidden_dim // num_heads), hidden_dim), dtype=jax.numpy.float32),
            },
            'feed_forward': {
                'in_weight': kaiming_normal_init(block_key, (hidden_dim, ff_dim), dtype=jax.numpy.float32),
                'out_weight': xavier_uniform_init(block_key, (ff_dim, hidden_dim), dtype=jax.numpy.float32),
            }
        }
        learnable_params['transformer_blocks'].append(block_params)

    static_config = {
        'scale': float((hidden_dim // num_heads) ** -0.5),
        'n_heads': int(num_heads),
        'mask': jax.numpy.triu(jax.numpy.ones((seq_len, seq_len), dtype=jax.numpy.bool), k=1),
        'pos': jax.numpy.arange(seq_len, dtype=jax.numpy.uint16)
    }

    return learnable_params, static_config