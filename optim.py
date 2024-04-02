import jax

def create_adam_state(params, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    return {"step": 0, "learning_rate": learning_rate, "beta_1": beta_1, "beta_2": beta_2, "epsilon": epsilon, "m": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), params), "v": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), params)}

def apply_adam_optimizer(learnable_params, adam_state, grads):
    adam_state['step'] += 1
    adam_state['m'] = jax.tree_util.tree_map(lambda m, g: (adam_state['beta_1'] * m) + (1 - adam_state['beta_1']) * g, adam_state['m'], grads)
    adam_state['v'] = jax.tree_util.tree_map(lambda v, g: (adam_state['beta_2'] * v) + (1 - adam_state['beta_2']) * (g ** 2), adam_state['v'], grads)
    m_corr = jax.tree_util.tree_map(lambda m: m / (1 - adam_state['beta_1'] ** adam_state['step']), adam_state['m'])
    v_corr = jax.tree_util.tree_map(lambda v: v / (1 - adam_state['beta_2'] ** adam_state['step']), adam_state['v'])
    updates = jax.tree_util.tree_map(lambda m, v: adam_state['learning_rate'] * m / (jax.numpy.sqrt(v) + adam_state['epsilon']), m_corr, v_corr)
    learnable_params = jax.tree_util.tree_map(lambda p, u: p - u, learnable_params, updates)
    return learnable_params, adam_state

def create_sm3_state(params, learning_rate=1e-3, momentum=0.9):
    return {"step": 0, "learning_rate": learning_rate, "momentum": momentum, "grad_avg": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), params)}

def apply_sm3_optimizer(learnable_params, sm3_state, grads):
    sm3_state["step"] += 1
    learnable_params = jax.tree_util.tree_map(lambda p, g, avg: p - sm3_state['learning_rate'] * (sm3_state["momentum"] * avg + (1 - sm3_state["momentum"]) * g), learnable_params, grads, sm3_state["grad_avg"])
    sm3_state["grad_avg"] = jax.tree_util.tree_map(lambda g, avg: sm3_state["momentum"] * avg + (1 - sm3_state["momentum"]) * g, grads, sm3_state["grad_avg"])
    return learnable_params, sm3_state

def create_rmsprop_state(params, learning_rate=1e-3, decay_rate=0.9, epsilon=1e-8):
    return {"step": 0, "learning_rate": learning_rate, "decay_rate": decay_rate, "epsilon": epsilon, "grad_sq_avg": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), params)}

def apply_rmsprop_optimizer(learnable_params, rmsprop_state, grads):
    rmsprop_state["step"] += 1
    learnable_params = jax.tree_util.tree_map(lambda p, g, avg: p - rmsprop_state['learning_rate'] * g / (jax.numpy.sqrt(rmsprop_state["decay_rate"] * avg + (1 - rmsprop_state["decay_rate"]) * jax.numpy.square(g)) + rmsprop_state["epsilon"]), learnable_params, grads, rmsprop_state["grad_sq_avg"])
    rmsprop_state["grad_sq_avg"] = jax.tree_util.tree_map(lambda g, avg: rmsprop_state["decay_rate"] * avg + (1 - rmsprop_state["decay_rate"]) * jax.numpy.square(g), grads, rmsprop_state["grad_sq_avg"])
    return learnable_params, rmsprop_state