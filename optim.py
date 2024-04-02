import jax

def create_adam_state(params, learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    return {"step": 0, "learning_rate": learning_rate, "beta_1": beta_1, "beta_2": beta_2, "epsilon": epsilon, "m": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), params), "v": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), params)}

def apply_adam_optimizer(learnable_params, adam_state, grads, warmup_steps, total_steps):
    adam_state['step'] += 1
    adam_state['m'] = jax.tree_util.tree_map(lambda m, g: (adam_state['beta_1'] * m) + (1 - adam_state['beta_1']) * g, adam_state['m'], grads)
    adam_state['v'] = jax.tree_util.tree_map(lambda v, g: (adam_state['beta_2'] * v) + (1 - adam_state['beta_2']) * (g ** 2), adam_state['v'], grads)
    m_corr = jax.tree_util.tree_map(lambda m: m / (1 - adam_state['beta_1'] ** adam_state['step']), adam_state['m'])
    v_corr = jax.tree_util.tree_map(lambda v: v / (1 - adam_state['beta_2'] ** adam_state['step']), adam_state['v'])
    learning_rate = jax.lax.cond(adam_state['step'] <= warmup_steps, lambda _: adam_state['learning_rate'] * (adam_state['step'] / warmup_steps), lambda _: 0.5 * adam_state['learning_rate'] * (1 + jax.numpy.cos(jax.numpy.pi * (jax.numpy.minimum(adam_state['step'] / total_steps, 1.0)))), None)
    updates = jax.tree_util.tree_map(lambda m, v: learning_rate * m / (jax.numpy.sqrt(v) + adam_state['epsilon']), m_corr, v_corr)
    learnable_params = jax.tree_util.tree_map(lambda p, u: p - u, learnable_params, updates)
    return learnable_params, adam_state, learning_rate