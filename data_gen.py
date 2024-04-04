import json
import gzip
import tqdm
import pickle
import tiktoken
from tiktoken.load import load_tiktoken_bpe

# gzip -c dolma/tokenized_books_and_wiki.pkl | split -b 1GB - dolma/tokenized_books_and_wiki.pkl.gz.
# cat dolma/tokenized_books_and_wiki.pkl.gz.* | gzip -d > dolma/tokenized_books_and_wiki.pkl

# screen -L -S data_gen -t data_gen bash -c 'cd /home/markusheimerl/transformer && /bin/python3 /home/markusheimerl/transformer/data_gen.py'

tokenizer = tiktoken.Encoding(
    name="cl100k_tokenizer",
    pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    mergeable_ranks=load_tiktoken_bpe("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
    special_tokens={"<|system|>": 100257, "<|user|>": 100258, "<|assistant|>": 100259, "<|endoftext|>": 100260}
)

books_and_wiki=[]

for i in tqdm.tqdm(range(3)):
    with gzip.open(f"dolma/books-000{i}.json.gz") as f:
        for line in f:
            books_and_wiki.append(json.loads(line.decode('utf-8').strip())["text"] + " <|endoftext|>")

for i in tqdm.tqdm(range(2)):
    with gzip.open(f"dolma/en_simple_wiki_v0-000{i}.json.gz") as f:
        for line in f:
            books_and_wiki.append(json.loads(line.decode('utf-8').strip())["text"] + " <|endoftext|>")

tokenized_books_and_wiki = tokenizer.encode_batch(books_and_wiki, num_threads=16, allowed_special="all")
tokenized_books_and_wiki = [item for sublist in tokenized_books_and_wiki for item in sublist]

with open("dolma/tokenized_books_and_wiki.pkl", "wb") as file:
    pickle.dump(tokenized_books_and_wiki, file)