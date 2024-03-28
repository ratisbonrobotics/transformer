import jax
import tqdm
import tiktoken
import random
from model import language_model
from tiktoken.load import load_tiktoken_bpe

# JAX_PLATFORMS='' /bin/python3 /home/markusheimerl/transformer/infer.py

# Load the checkpoint and tokenizer
checkpoint = jax.numpy.load("checkpoint_232390.npz", allow_pickle=True)
learnable_params = checkpoint["learnable_params"].item()
static_config = {
    "pos": checkpoint["static_config_pos"],
    "n_heads": checkpoint["static_config_n_heads"].item(),
    "scale": checkpoint["static_config_scale"].item()
}

tokenizer = tiktoken.Encoding(
    name="cl100k_tokenizer",
    pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    mergeable_ranks=load_tiktoken_bpe("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
    special_tokens={"<|system|>": 100257, "<|user|>": 100258, "<|assistant|>": 100259, "<|endoftext|>": 100260}
)

# Define the inference function
def generate_text(key, prompt, max_length=100, temperature=0.7, top_p=0.9):
    prompt_tokens = tokenizer.encode(prompt, allowed_special="all")
    token_ids = jax.numpy.array(prompt_tokens, dtype=jax.numpy.uint32)
    
    for _ in tqdm.tqdm(range(max_length)):
        logits = language_model(learnable_params, token_ids[None, :], static_config['pos'][:token_ids.shape[0]], static_config['n_heads'], static_config['scale'])
        logits = logits[0, -1] / temperature
        probs = jax.nn.softmax(logits)
        # Sort the probabilities in descending order
        sorted_probs = jax.numpy.sort(probs)[::-1]
        # Compute the cumulative sum of probabilities
        cumulative_probs = jax.numpy.cumsum(sorted_probs)
        # Find the index where the cumulative probability exceeds top_p
        top_p_index = jax.numpy.argmax(cumulative_probs > top_p)
        # Get the indices of the tokens with probabilities greater than the top_p threshold
        top_p_indices = jax.numpy.argsort(probs)[-top_p_index:]
        # Create a multinomial probability distribution using the top_p tokens
        top_p_probs = probs[top_p_indices]
        top_p_probs /= jax.numpy.sum(top_p_probs)
        # Sample the next token from the multinomial distribution
        next_token = jax.random.choice(key, top_p_indices, p=top_p_probs)
        token_ids = jax.numpy.append(token_ids, next_token)
    
    return tokenizer.decode(token_ids)

# Infer model, starting from prompt
prompt = "[SYSTEM] You are an AI assistant. You will be given a task. You must generate a detailed and long answer. [USER] What happens next in this paragraph? She then rubs a needle on a cotton ball then pushing it onto a pencil and wrapping thread around it. She then holds up a box of a product and then pouring several liquids into a bowl. she Choose your answer from: A. adds saucepan and shakes up the product in a grinder. B. pinches the thread to style a cigarette, and then walks away. C. then dips the needle in ink and using the pencil to draw a design on her leg, rubbing it off with a rag in the end. D. begins to style her hair and cuts it several times before parting the ends of it to show the hairstyle she has created. [ASSISTANT]"
generated_text = generate_text(jax.random.PRNGKey(random.randint(0, 10000)), prompt)
print(generated_text)