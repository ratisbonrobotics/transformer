import jax

def feed_forward(params, x):
    x = jax.numpy.dot(x, params['in_weight'])
    x = jax.nn.gelu(x, approximate=True)
    x = jax.numpy.dot(x, params['out_weight'])
    return x

def attention(params, x, mask):
    batch_size, seq_len, _ = x.shape
    # Linear transformations
    q = jax.numpy.dot(x, params['q_linear']).reshape(batch_size, seq_len, params['n_heads'], -1).transpose(0, 2, 1, 3)
    k = jax.numpy.dot(x, params['k_linear']).reshape(batch_size, seq_len, params['n_heads'], -1).transpose(0, 2, 1, 3)
    v = jax.numpy.dot(x, params['v_linear']).reshape(batch_size, seq_len, params['n_heads'], -1).transpose(0, 2, 1, 3)
    # Compute attention scores
    scores = jax.numpy.matmul(q, k.transpose(0, 1, 3, 2)) * params['scale']
    scores = jax.numpy.where(mask[:seq_len, :seq_len], -jax.numpy.inf, scores)
    scores = jax.nn.softmax(scores, axis=-1)
    # Compute output
    output = jax.numpy.matmul(scores, v)
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    output = jax.numpy.dot(output, params['o_linear'])
    return output

def transformer_block(params, x, mask):
    r = attention(params['attention'], layer_norm(x, params['attention_norm_scale'], params['attention_norm_bias']), mask)
    h = x + r
    r = feed_forward(params['feed_forward'], layer_norm(h, params['ffn_norm_scale'], params['ffn_norm_bias']))
    out = h + r
    return out

def layer_norm(x, scale, bias):
    mean = jax.numpy.mean(x, axis=-1, keepdims=True)
    variance = jax.numpy.mean(jax.numpy.square(x - mean), axis=-1, keepdims=True)
    normalized = (x - mean) * jax.lax.rsqrt(variance + 1e-5)
    return normalized * scale + bias

def language_model(params, token_ids, static_config):
    x = params['tok_emb'][token_ids] + params['pos_emb'][static_config['pos']]
    x = layer_norm(x, params['pos_norm_scale'], params['pos_norm_bias'])
    for block_params in params['transformer_blocks']:
        x = transformer_block(block_params, x, static_config['mask'])
    x = layer_norm(x, params['out_norm_scale'], params['out_norm_bias'])
    return jax.numpy.dot(x, params['out_linear_weight'])

def init_params(vocab_size=32768, seq_len=2048, num_blocks=16, num_heads=8, hidden_dim=768, ff_dim=2048, rng_key=jax.random.PRNGKey(0)):
    rng_key, subkey = jax.random.split(rng_key)
    learnable_params = {
        'tok_emb': jax.random.normal(subkey, (vocab_size, hidden_dim)) * 0.02,
        'pos_emb': jax.random.normal(subkey, (vocab_size, hidden_dim)) * 0.02,
        'pos': jax.numpy.arange(seq_len),
        'pos_norm_scale': jax.numpy.ones(hidden_dim),
        'pos_norm_bias': jax.numpy.zeros(hidden_dim),
        'mask': jax.numpy.triu(jax.numpy.ones((seq_len, seq_len)), k=1).astype(bool),
        'transformer_blocks': [],
        'out_norm_scale': jax.numpy.ones(hidden_dim),
        'out_norm_bias': jax.numpy.zeros(hidden_dim),
        'out_linear_weight': jax.nn.initializers.glorot_uniform()(rng_key, (hidden_dim, vocab_size)),
    }
    for _ in range(num_blocks):
        rng_key, block_key = jax.random.split(rng_key)
        block_params = {
            'attention': {
                'n_heads': num_heads,
                'scale': (hidden_dim // num_heads) ** -0.5,
                'q_linear': jax.nn.initializers.glorot_uniform()(block_key, (hidden_dim, num_heads * (hidden_dim // num_heads))),
                'k_linear': jax.nn.initializers.glorot_uniform()(block_key, (hidden_dim, num_heads * (hidden_dim // num_heads))),
                'v_linear': jax.nn.initializers.glorot_uniform()(block_key, (hidden_dim, num_heads * (hidden_dim // num_heads))),
                'o_linear': jax.nn.initializers.glorot_uniform()(block_key, (num_heads * (hidden_dim // num_heads), hidden_dim)),
            },
            'feed_forward': {
                'in_weight': jax.nn.initializers.he_normal()(block_key, (hidden_dim, ff_dim)),
                'out_weight': jax.nn.initializers.glorot_uniform()(block_key, (ff_dim, hidden_dim)),
            },
            'attention_norm_scale': jax.numpy.ones(hidden_dim),
            'attention_norm_bias': jax.numpy.zeros(hidden_dim),
            'ffn_norm_scale': jax.numpy.ones(hidden_dim),
            'ffn_norm_bias': jax.numpy.zeros(hidden_dim),
        }
        learnable_params['transformer_blocks'].append(block_params)

    static_config = {
        'mask': jax.numpy.triu(jax.numpy.ones((seq_len, seq_len)), k=1).astype(bool),
        'pos': jax.numpy.arange(seq_len)
    }

    return learnable_params, static_config