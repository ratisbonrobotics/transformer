from datasets import load_dataset
import pickle

# gzip -c open_orca_cache.pkl | split -b 1GB - open_orca_cache.pkl.gz.
# cat open_orca_cache.pkl.gz.* | gzip -d > open_orca_cache.pkl

# Load the dataset
dataset = load_dataset("Open-Orca/OpenOrca")

# Function to apply modification
def apply_modification(row):
    row["data"] = "[SYSTEM] " + row["system_prompt"] + " [USER] " + row["question"] + " [ASSISTANT] " + row["response"]
    return row

modified_dataset = dataset.map(apply_modification, num_proc=8, remove_columns=["id", "system_prompt", "question", "response"])

with open("open_orca.pkl", "wb") as file:
    pickle.dump(modified_dataset["train"]["data"], file)

