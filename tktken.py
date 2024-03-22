import tiktoken
from tiktoken.load import load_tiktoken_bpe
import pickle
import random

tokenizer = tiktoken.Encoding(
    name="cl100k_tokenizer",
    pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    mergeable_ranks=load_tiktoken_bpe("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
    special_tokens={"[SYSTEM]": 100257, "[USER]": 100258, "[ASSISTANT]": 100259}
)

print(tokenizer.n_vocab)

with open("open_orca.pkl", "rb") as file:
    openorca = pickle.load(file)

idx = random.randint(0, len(openorca) - 17)

assert tokenizer.decode_batch(tokenizer.encode_batch(openorca[idx:idx+16], allowed_special="all")) == openorca[idx:idx+16]

print(tokenizer.encode_batch(openorca[idx:idx+16], allowed_special="all"))
print(tokenizer.decode_batch(tokenizer.encode_batch(openorca[idx:idx+16], allowed_special="all")))