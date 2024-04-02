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

def create_adafactor_state(params, learning_rate=1e-3, beta1=0.0, decay_rate=0.8, epsilon1=1e-30, epsilon2=1e-3):
    def init_adafactor_state(p):
        return {
            "exp_avg": jax.numpy.zeros_like(p),
            "exp_avg_sq_row": jax.numpy.zeros((p.shape[0],)) if p.ndim == 2 else None,
            "exp_avg_sq_col": jax.numpy.zeros((p.shape[1],)) if p.ndim == 2 else None,
        }

    adafactor_state = jax.tree_util.tree_map(init_adafactor_state, params)
    adafactor_state["step"] = 0
    adafactor_state["learning_rate"] = learning_rate
    adafactor_state["beta1"] = beta1
    adafactor_state["decay_rate"] = decay_rate
    adafactor_state["epsilon1"] = epsilon1
    adafactor_state["epsilon2"] = epsilon2
    return adafactor_state

def apply_adafactor_optimizer(learnable_params, adafactor_state, grads):
    adafactor_state["step"] += 1

    def update_param(p, g, state):
        state["exp_avg"] = adafactor_state["beta1"] * state["exp_avg"] + (1 - adafactor_state["beta1"]) * g

        def update_2d(_):
            state["exp_avg_sq_row"] = adafactor_state["decay_rate"] * state["exp_avg_sq_row"] + (1 - adafactor_state["decay_rate"]) * jax.numpy.mean(jax.numpy.square(g), axis=1)
            state["exp_avg_sq_col"] = adafactor_state["decay_rate"] * state["exp_avg_sq_col"] + (1 - adafactor_state["decay_rate"]) * jax.numpy.mean(jax.numpy.square(g), axis=0)
            exp_avg_sq_row = jax.numpy.maximum(state["exp_avg_sq_row"], adafactor_state["epsilon2"])
            exp_avg_sq_col = jax.numpy.maximum(state["exp_avg_sq_col"], adafactor_state["epsilon2"])
            row_factor = jax.numpy.sqrt(exp_avg_sq_row) / (jax.numpy.sqrt(exp_avg_sq_col) + adafactor_state["epsilon1"])
            col_factor = jax.numpy.sqrt(exp_avg_sq_col) / (jax.numpy.sqrt(exp_avg_sq_row) + adafactor_state["epsilon1"])
            update = state["exp_avg"] / (jax.numpy.outer(row_factor, col_factor) + adafactor_state["epsilon1"])
            return p - adafactor_state["learning_rate"] * update

        def update_other(_):
            update = state["exp_avg"] / (jax.numpy.sqrt(jax.numpy.mean(jax.numpy.square(state["exp_avg"]))) + adafactor_state["epsilon1"])
            return p - adafactor_state["learning_rate"] * update

        p = jax.lax.cond(p.ndim == 2, update_2d, update_other, None)
        return p, state

    learnable_params, adafactor_state = jax.tree_util.tree_map(update_param, learnable_params, grads, adafactor_state)
    return learnable_params, adafactor_state