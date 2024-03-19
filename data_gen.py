from datasets import load_dataset
import pickle

# cat open_orca_cache.pkl.gz.* | gzip -d > open_orca_cache.pkl

# Load the dataset
dataset = load_dataset("Open-Orca/OpenOrca", cache_dir="/dev/shm/")

# Function to apply modification
def apply_modification(row):
    row["data"] = "[USER] " + row["question"].lower() + " [ASSISTANT] " + row["response"].lower()
    return row

modified_dataset = dataset.map(apply_modification, num_proc=8, remove_columns=["id", "system_prompt", "question", "response"])

with open("open_orca.pkl", "wb") as file:
    pickle.dump(modified_dataset["train"]["data"][:10000], file)

