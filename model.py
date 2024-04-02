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
    for block_params in params['transformer_blocks']:
        x = transformer_block(block_params, x, mask, n_heads, scale)
    return jax.numpy.dot(simple_rms_norm(x), params['out_linear'])

def init_params(vocab_size, seq_len, num_blocks=16, num_heads=8, hidden_dim=2048, ff_dim=8192, dtype=jax.numpy.float32, rng_key=jax.random.key(0)):
    xavier_uniform_init = jax.nn.initializers.glorot_uniform(dtype=dtype)
    kaiming_normal_init = jax.nn.initializers.he_normal(dtype=dtype)
    
    rng_key, tok_emb_key, pos_emb_key, out_linear_key = jax.random.split(rng_key, 4)
    
    learnable_params = {
        'tok_emb': jax.random.normal(tok_emb_key, (vocab_size, hidden_dim), dtype=dtype) * 0.02,
        'pos_emb': jax.random.normal(pos_emb_key, (vocab_size, hidden_dim), dtype=dtype) * 0.02,
        'transformer_blocks': [],
        'out_linear': xavier_uniform_init(out_linear_key, (hidden_dim, vocab_size), dtype=dtype),
    }
    
    block_keys = jax.random.split(rng_key, num_blocks)
    
    for block_key in block_keys:
        q_linear_key, k_linear_key, v_linear_key, o_linear_key, in_weight_key, out_weight_key = jax.random.split(block_key, 6)
        
        block_params = {
            'attention': {
                'q_linear': xavier_uniform_init(q_linear_key, (hidden_dim, num_heads * (hidden_dim // num_heads)), dtype=dtype),
                'k_linear': xavier_uniform_init(k_linear_key, (hidden_dim, num_heads * (hidden_dim // num_heads)), dtype=dtype),
                'v_linear': xavier_uniform_init(v_linear_key, (hidden_dim, num_heads * (hidden_dim // num_heads)), dtype=dtype),
                'o_linear': xavier_uniform_init(o_linear_key, (num_heads * (hidden_dim // num_heads), hidden_dim), dtype=dtype),
            },
            'feed_forward': {
                'in_weight': kaiming_normal_init(in_weight_key, (hidden_dim, ff_dim), dtype=dtype),
                'out_weight': xavier_uniform_init(out_weight_key, (ff_dim, hidden_dim), dtype=dtype),
            }
        }
        learnable_params['transformer_blocks'].append(block_params)
    
    static_config = {
        'scale': float((hidden_dim // num_heads) ** -0.5),
        'n_heads': int(num_heads),
        'mask': jax.numpy.triu(jax.numpy.ones((seq_len, seq_len), dtype=jax.numpy.bool), k=1),
        'pos': jax.numpy.arange(1, seq_len + 1, dtype=jax.numpy.uint16)
    }
    
    return learnable_params, static_config