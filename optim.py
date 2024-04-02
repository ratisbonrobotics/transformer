import jax

def create_adam_state(learnable_params, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    return {"step": 0, "learning_rate": learning_rate, "beta_1": beta_1, "beta_2": beta_2, "epsilon": epsilon, "m": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), learnable_params), "v": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), learnable_params)}

def apply_adam_optimizer(learnable_params, adam_state, grads):
    adam_state['step'] += 1
    adam_state['m'] = jax.tree_util.tree_map(lambda m, g: (adam_state['beta_1'] * m) + (1 - adam_state['beta_1']) * g, adam_state['m'], grads)
    adam_state['v'] = jax.tree_util.tree_map(lambda v, g: (adam_state['beta_2'] * v) + (1 - adam_state['beta_2']) * (g ** 2), adam_state['v'], grads)
    m_corr = jax.tree_util.tree_map(lambda m: m / (1 - adam_state['beta_1'] ** adam_state['step']), adam_state['m'])
    v_corr = jax.tree_util.tree_map(lambda v: v / (1 - adam_state['beta_2'] ** adam_state['step']), adam_state['v'])
    updates = jax.tree_util.tree_map(lambda m, v: adam_state['learning_rate'] * m / (jax.numpy.sqrt(v) + adam_state['epsilon']), m_corr, v_corr)
    learnable_params = jax.tree_util.tree_map(lambda p, u: p - u, learnable_params, updates)
    return learnable_params, adam_state

def create_rmsprop_state(learnable_params, learning_rate=1e-3, decay_rate=0.9, epsilon=1e-8):
    return {"step": 0, "learning_rate": learning_rate, "decay_rate": decay_rate, "epsilon": epsilon, "grad_sq_avg": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), learnable_params)}

def apply_rmsprop_optimizer(learnable_params, rmsprop_state, grads):
    rmsprop_state["step"] += 1
    learnable_params = jax.tree_util.tree_map(lambda p, g, avg: p - rmsprop_state['learning_rate'] * g / (jax.numpy.sqrt(rmsprop_state["decay_rate"] * avg + (1 - rmsprop_state["decay_rate"]) * jax.numpy.square(g)) + rmsprop_state["epsilon"]), learnable_params, grads, rmsprop_state["grad_sq_avg"])
    rmsprop_state["grad_sq_avg"] = jax.tree_util.tree_map(lambda g, avg: rmsprop_state["decay_rate"] * avg + (1 - rmsprop_state["decay_rate"]) * jax.numpy.square(g), grads, rmsprop_state["grad_sq_avg"])
    return learnable_params, rmsprop_state

def create_adagrad_state(learnable_params, learning_rate=1e-3, epsilon=1e-8):
    return {"step": 0, "learning_rate": learning_rate, "epsilon": epsilon, "grad_sq_sum": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), learnable_params)}

def apply_adagrad_optimizer(learnable_params, adagrad_state, grads):
    adagrad_state["step"] += 1
    adagrad_state["grad_sq_sum"] = jax.tree_util.tree_map(lambda g, s: s + jax.numpy.square(g), grads, adagrad_state["grad_sq_sum"])
    learnable_params = jax.tree_util.tree_map(lambda p, g, s: p - adagrad_state['learning_rate'] * g / (jax.numpy.sqrt(s) + adagrad_state["epsilon"]), learnable_params, grads, adagrad_state["grad_sq_sum"])
    return learnable_params, adagrad_state

def create_sgd_state(_, learning_rate=1e-3):
    return {"step": 0, "learning_rate": learning_rate}

def apply_sgd_optimizer(learnable_params, sgd_state, grads):
    sgd_state["step"] += 1
    learnable_params = jax.tree_util.tree_map(lambda p, g: p - sgd_state['learning_rate'] * g, learnable_params, grads)
    return learnable_params, sgd_state