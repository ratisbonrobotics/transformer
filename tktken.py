import tiktoken
import pickle

with open("open_orca.pkl", "rb") as file:
    loaded_dialogs = pickle.load(file)

print(loaded_dialogs[0])