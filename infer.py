import jax
import tqdm
import tiktoken
from model import language_model
from tiktoken.load import load_tiktoken_bpe

# JAX_PLATFORMS='' /bin/python3 /home/markusheimerl/transformer/infer.py

# Load the checkpoint and tokenizer
checkpoint = jax.numpy.load("checkpoint_8_123030.npz", allow_pickle=True)
learnable_params = checkpoint["learnable_params"].item()
static_config = {
    "pos": checkpoint["static_config_pos"],
    "mask": checkpoint["static_config_mask"],
    "n_heads": checkpoint["static_config_n_heads"].item(),
    "scale": checkpoint["static_config_scale"].item()
}

tokenizer = tiktoken.Encoding(
    name="cl100k_tokenizer",
    pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    mergeable_ranks=load_tiktoken_bpe("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
    special_tokens={"[SYSTEM]": 100257, "[USER]": 100258, "[ASSISTANT]": 100259}
)

# Define the inference function
def generate_text(prompt, max_length=100, temperature=0.7):
    prompt_tokens = tokenizer.encode(prompt, allowed_special="all")
    token_ids = jax.numpy.array(prompt_tokens, dtype=jax.numpy.uint32)
    
    for _ in tqdm.tqdm(range(max_length)):
        logits = language_model(learnable_params, token_ids[None, :], static_config['pos'][:token_ids.shape[0]], static_config['mask'], static_config['n_heads'], static_config['scale'])
        logits = logits[0, -1] / temperature
        probs = jax.nn.softmax(logits)
        next_token = jax.numpy.argmax(probs)
        token_ids = jax.numpy.append(token_ids, next_token)
    
    return tokenizer.decode(token_ids)

# Then call your function the same way
prompt = "[SYSTEM] You are an AI assistant. You will be given a task. You must generate a detailed and long answer. [USER] What happens next in this paragraph? She then rubs a needle on a cotton ball then pushing it onto a pencil and wrapping thread around it. She then holds up a box of a product and then pouring several liquids into a bowl. she Choose your answer from: A. adds saucepan and shakes up the product in a grinder. B. pinches the thread to style a cigarette, and then walks away. C. then dips the needle in ink and using the pencil to draw a design on her leg, rubbing it off with a rag in the end. D. begins to style her hair and cuts it several times before parting the ends of it to show the hairstyle she has created. [ASSISTANT]"
generated_text = generate_text(prompt, max_length=15, temperature=0.7)
print(generated_text)