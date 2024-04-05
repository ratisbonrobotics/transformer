import os
import json
import gzip
import tqdm
import pickle
import tiktoken
from tiktoken.load import load_tiktoken_bpe

def process_file(file_path : str):
    text_data = []
    with gzip.open(file_path) as f:
        for line in tqdm.tqdm(f):
            text_data.append(json.loads(line.decode('utf-8'))["text"])
    tokenized_text_data = tokenizer.encode_batch(text_data, num_threads=128, allowed_special="all")
    output_file = file_path.replace(".json.gz", "-tok.pkl")
    with open(output_file, "wb") as file:
        for data in tokenized_text_data:
            pickle.dump(data, file)

tokenizer = tiktoken.Encoding(
    name="cl100k_tokenizer",
    pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    mergeable_ranks=load_tiktoken_bpe("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
    special_tokens={"<|system|>": 100257, "<|user|>": 100258, "<|assistant|>": 100259, "<|endoftext|>": 100260}
)

for file_name in os.listdir("dolma"):
    if file_name.endswith(".json.gz"):
        file_path = os.path.join("dolma", file_name)
        process_file(file_path)