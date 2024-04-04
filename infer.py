import jax
import tiktoken
import random
from model import language_model
from tiktoken.load import load_tiktoken_bpe

# JAX_PLATFORMS='' /bin/python3 /home/markusheimerl/transformer/infer.py

# Load the checkpoint and tokenizer
checkpoint = jax.numpy.load("checkpoint_48106.npz", allow_pickle=True)
learnable_params = checkpoint["learnable_params"].item()
static_config = {
    "mask": checkpoint["static_config_mask"]
}

tokenizer = tiktoken.Encoding(
    name="cl100k_tokenizer",
    pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    mergeable_ranks=load_tiktoken_bpe("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
    special_tokens={"<|system|>": 100257, "<|user|>": 100258, "<|assistant|>": 100259, "<|endoftext|>": 100260}
)

# Define the inference function
def generate_token(key, token_ids, temperature=0.7, top_k=40):
    logits = language_model(learnable_params, token_ids[None, :], static_config['mask'])
    logits = logits[0, -1] / temperature
    top_k_indices = jax.numpy.argsort(logits)[-top_k:]
    top_k_logits = logits[top_k_indices]
    top_k_probs = jax.nn.softmax(top_k_logits)
    next_token = jax.random.choice(key, top_k_indices, p=top_k_probs)
    return next_token

generate_text_jit = jax.jit(generate_token)

# Infer model, starting from prompt
prompt = "<|system|> You are an AI assistant. You will be given a task. You must generate a detailed and long answer. <|user|> What happens next in this paragraph? She then rubs a needle on a cotton ball then pushing it onto a pencil and wrapping thread around it. She then holds up a box of a product and then pouring several liquids into a bowl. she Choose your answer from: A. adds saucepan and shakes up the product in a grinder. B. pinches the thread to style a cigarette, and then walks away. C. then dips the needle in ink and using the pencil to draw a design on her leg, rubbing it off with a rag in the end. D. begins to style her hair and cuts it several times before parting the ends of it to show the hairstyle she has created. <|assistant|>"
print(prompt, end="", flush=True)
key = jax.random.key(random.randint(0, 2**16-1))
token_ids = jax.numpy.array(tokenizer.encode(prompt, allowed_special="all"), dtype=jax.numpy.uint32)
for _ in range(50):
    key, round_key = jax.random.split(key)
    next_token = generate_token(round_key, token_ids)
    token_ids = jax.numpy.append(token_ids, next_token)
    print(tokenizer.decode([next_token]), end="", flush=True)

print("")