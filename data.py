import os
import jax
import json
import gzip
import tqdm
import time
import pickle
import random
import tiktoken
import itertools
from tiktoken.load import load_tiktoken_bpe

# aria2c -x 16 https://storage.googleapis.com/xvit-415020_dolma_tokenized/books-0000-tok.pkl
# aria2c -x 16 https://storage.googleapis.com/xvit-415020_dolma_tokenized/books-0001-tok.pkl
# aria2c -x 16 https://storage.googleapis.com/xvit-415020_dolma_tokenized/books-0002-tok.pkl

# aria2c -x 16 https://storage.googleapis.com/xvit-415020_dolma_tokenized/en_simple_wiki_v0-0000-tok.pkl
# aria2c -x 16 https://storage.googleapis.com/xvit-415020_dolma_tokenized/en_simple_wiki_v0-0001-tok.pkl

class TextDataset:
    def __init__(self, file_paths, sequence_length=2048):
        
        tokenizer = tiktoken.Encoding(
            name="cl100k_tokenizer",
            pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
            mergeable_ranks=load_tiktoken_bpe("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
            special_tokens={"<|system|>": 100257, "<|user|>": 100258, "<|assistant|>": 100259, "<|endoftext|>": 100260}
        )

        self.vocab_size = tokenizer.n_vocab
        self.sequence_length = sequence_length

        self.text_data = []

        start_time = time.time()
        for file_path in file_paths:
            print(f"Loading {file_path}...")
            with open(file_path , 'rb') as f:
                while True:
                    try:
                        entry = pickle.load(f)
                        self.text_data.append(entry + [100260])
                    except EOFError:
                        break

        print(f"Data loading time: {(time.time() - start_time):.4f}s")
        random.shuffle(self.text_data)
        self.text_data = list(itertools.chain.from_iterable(self.text_data))
        print(f"Total number of tokens per epoch: {len(self.text_data)}")

    def __len__(self):
        return (len(self.text_data) - (self.sequence_length + 1)) // self.sequence_length

    def __getitem__(self, idx):
        idx = idx * self.sequence_length
        inputs = jax.numpy.array(self.text_data[idx : idx + self.sequence_length], dtype=jax.numpy.uint32)
        labels = jax.numpy.array(self.text_data[idx + 1 : idx + self.sequence_length + 1], dtype=jax.numpy.uint32)
        return inputs, labels

def data_gen():
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

    for file_name in os.listdir("."):
        if file_name.endswith(".json.gz"):
            file_path = os.path.join(".", file_name)
            process_file(file_path)
