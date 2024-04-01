import os
import jax
import glob
import tqdm
import wandb
import random
from model import video_model, init_params

# screen -L -S train -t train bash -c 'cd /home/markusheimerl/transformer && /bin/python3 /home/markusheimerl/transformer/train.py'

# Constants
NUM_EPOCHS = 10
BATCH_SIZE = 2
WARMUP_STEPS = 5
WANDB = False

def create_adam_state(params, learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    return {"step": 0, "learning_rate": learning_rate, "beta_1": beta_1, "beta_2": beta_2, "epsilon": epsilon, "m": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), params), "v": jax.tree_util.tree_map(lambda p: jax.numpy.zeros_like(p), params)}

class VideoDataset:
    def __init__(self, folder_path, height_seq_len=30, width_seq_len=40, cache_file="video_data_cache.npz"):
        self.vocab_size = 8 * 16 * 16 * 3
        self.height_seq_len = height_seq_len
        self.width_seq_len = width_seq_len
        
        if os.path.exists(cache_file):
            self.video_data = jax.numpy.load(cache_file)["video_data"]
        else:
            npz_files = glob.glob(os.path.join(folder_path, "*.npz"))
            loaded_video_data = []
            
            for npz_file in npz_files:
                data = jax.numpy.load(npz_file)["patches"]
                data = data[:, :self.height_seq_len, :self.width_seq_len, :, :, :, :]
                loaded_video_data.append(data)
            
            loaded_video_data = jax.numpy.concatenate(loaded_video_data, axis=0)
            self.video_data = loaded_video_data.reshape(loaded_video_data.shape[0], loaded_video_data.shape[1], loaded_video_data.shape[2], -1)
            jax.numpy.savez(cache_file, video_data=self.video_data)
    
    def __len__(self):
        return len(self.video_data) // 2
    
    def __getitem__(self, idx):
        idx *= 2
        inputs = self.video_data[idx]
        labels = self.video_data[idx + 1]
        return inputs, labels

# Create Dataset
train_dataset = VideoDataset("tensors/1d82d1f47b6aabf2d7bef9daa498ffc481fdd3bf63b128365ffa493c6df474ce_patches.npz", cache_file="1d82d1f47b6aabf2d7bef9daa498ffc481fdd3bf63b128365ffa493c6df474ce_patches_cached.npz")

# Create the model
random_seed = random.randint(0, 2**16-1)
learnable_params, static_config = init_params(vocab_size=train_dataset.vocab_size, height_seq_len=train_dataset.height_seq_len, width_seq_len=train_dataset.width_seq_len, rng_key=jax.random.PRNGKey(random_seed))
print(f"Total number of trainable parameters: {sum(jax.numpy.prod(jax.numpy.array(param.shape)).item() for param in jax.tree_util.tree_leaves(learnable_params))} - PRNG seed used for parameter initialization: {random_seed}")

# Create optimizer
adam_state = create_adam_state(learnable_params)

# Replicate model parameters across devices
static_config['height_pos'] = jax.device_put_replicated(static_config['height_pos'], jax.local_devices())
static_config['width_pos'] = jax.device_put_replicated(static_config['width_pos'], jax.local_devices())
learnable_params = jax.device_put_replicated(learnable_params, jax.local_devices())
adam_state = jax.device_put_replicated(adam_state, jax.local_devices())

# Define the loss function 
def loss_fn(learnable_params, inputs, labels, height_pos, width_pos, n_heads, scale):
    predictions = video_model(learnable_params, inputs, height_pos, width_pos, n_heads, scale)
    loss = jax.numpy.mean((predictions - labels) ** 2)
    # l2 loss
    loss += 1e-5 * jax.tree_util.tree_reduce(lambda x, y: x + y, jax.tree_util.tree_map(lambda p: jax.numpy.sum(p), jax.tree_util.tree_map(lambda p: jax.numpy.square(p), learnable_params)))
    return loss * 1024.0

# Define training step
def train_step(learnable_params, adam_state, inputs, labels, height_pos, width_pos, n_heads, scale, total_steps):
    # mixed precision
    learnable_params_bfloat16 = jax.tree_util.tree_map(lambda p: (p.astype(jax.numpy.bfloat16)), learnable_params)
    # calculate loss
    loss, grads = jax.value_and_grad(loss_fn)(learnable_params_bfloat16, inputs, labels, height_pos, width_pos, n_heads, scale)
    # gradient scaling
    grads = jax.tree_util.tree_map(lambda g: (g.astype(jax.numpy.float32) / 1024.0), grads)
    # exchange gradients
    grads = jax.lax.pmean(grads, axis_name='p')
    # adam optimizer
    adam_state['step'] += 1
    adam_state['m'] = jax.tree_util.tree_map(lambda m, g: (adam_state['beta_1'] * m) + (1 - adam_state['beta_1']) * g, adam_state['m'], grads)
    adam_state['v'] = jax.tree_util.tree_map(lambda v, g: (adam_state['beta_2'] * v) + (1 - adam_state['beta_2']) * (g ** 2), adam_state['v'], grads)
    m_corr = jax.tree_util.tree_map(lambda m: m / (1 - adam_state['beta_1'] ** adam_state['step']), adam_state['m'])
    v_corr = jax.tree_util.tree_map(lambda v: v / (1 - adam_state['beta_2'] ** adam_state['step']), adam_state['v'])
    learning_rate = jax.lax.cond(adam_state['step'] <= WARMUP_STEPS, lambda _: adam_state['learning_rate'] * (adam_state['step'] / WARMUP_STEPS), lambda _: adam_state['learning_rate'] * (1 - adam_state['step'] / total_steps) ** 2, None)
    updates = jax.tree_util.tree_map(lambda m, v: learning_rate * m / (jax.numpy.sqrt(v) + adam_state['epsilon']), m_corr, v_corr)
    learnable_params = jax.tree_util.tree_map(lambda p, u: p - u, learnable_params, updates)

    return learnable_params, adam_state, jax.lax.pmean(loss, axis_name='p') / 1024.0, learning_rate

jit_train_step = jax.pmap(train_step, static_broadcasted_argnums=(6,7,8), axis_name='p')

# Training loop
if WANDB: wandb.init(project="encoder")
for epoch in range(NUM_EPOCHS):
    indices = list(range(epoch, len(train_dataset), BATCH_SIZE * jax.local_device_count()))[:-1]
    random.shuffle(indices)
    with tqdm.tqdm(indices) as pbar:
        for batch_idx in pbar:
            batch_inputs, batch_labels = [], []
            for i in range(batch_idx, batch_idx + BATCH_SIZE * jax.local_device_count()):
                inputs, labels = train_dataset[i]
                batch_inputs.append(inputs)
                batch_labels.append(labels)
            
            # Split the batch across devices
            device_batch_inputs = jax.numpy.stack(batch_inputs, dtype=jax.numpy.uint32).reshape(jax.local_device_count(), BATCH_SIZE, train_dataset.height_seq_len * train_dataset.width_seq_len, train_dataset.vocab_size)
            device_batch_labels = jax.numpy.stack(batch_labels, dtype=jax.numpy.uint32).reshape(jax.local_device_count(), BATCH_SIZE, train_dataset.height_seq_len * train_dataset.width_seq_len, train_dataset.vocab_size)
            
            learnable_params, adam_state, loss, learning_rate = jit_train_step(learnable_params, adam_state, device_batch_inputs, device_batch_labels, static_config['height_pos'], static_config['width_pos'], static_config["n_heads"], static_config["scale"], len(indices) * NUM_EPOCHS)
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Loss: {float(jax.numpy.mean(loss)):.4f} - Learning Rate: {float(jax.numpy.mean(learning_rate)):.10f}")
            if WANDB: wandb.log({"loss": jax.numpy.mean(loss).item(), "learning_rate": jax.numpy.mean(learning_rate).item()})
    
    jax.numpy.savez(f"checkpoint_{adam_state['step'][0]}.npz",
        learnable_params=jax.tree_util.tree_map(lambda x: x[0], learnable_params),
        static_config_height_pos=jax.tree_util.tree_map(lambda x: x[0], static_config['height_pos']),
        static_config_width_pos=jax.tree_util.tree_map(lambda x: x[0], static_config['width_pos']),
        static_config_n_heads=static_config["n_heads"],
        static_config_scale=static_config["scale"]
    )
