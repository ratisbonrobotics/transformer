import jax
import tqdm
import tiktoken
from model import language_model
from tiktoken.load import load_tiktoken_bpe

# Load the checkpoint and tokenizer
checkpoint = jax.numpy.load("checkpoint_13_784.npz", allow_pickle=True)
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
    prompt_tokens = tokenizer.encode(prompt)
    token_ids = jax.numpy.array(prompt_tokens, dtype=jax.numpy.uint32)
    
    for _ in tqdm.tqdm(range(max_length)):
        logits = language_model(learnable_params, token_ids[None, :], static_config['pos'][:token_ids.shape[0]], static_config['mask'], static_config['n_heads'], static_config['scale'])
        logits = logits[0, -1] / temperature
        probs = jax.nn.softmax(logits)
        next_token = jax.random.categorical(jax.random.PRNGKey(0), probs)
        token_ids = jax.numpy.append(token_ids, next_token)
    
    return tokenizer.decode(token_ids)

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt, max_length=5, temperature=0.7)
print(generated_text)