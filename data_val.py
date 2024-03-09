import os
import pickle
import random
from collections import Counter

DATASET_PATH = "open_orca.pkl"

def load_dialogs():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATASET_PATH)

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Please make sure you have run the dialog collection script.")
        return []
    
    with open(file_path, "rb") as file:
        dialogs = pickle.load(file)
    return dialogs

def print_basic_statistics(dialogs):
    print(f"Total number of dialogs: {len(dialogs)}")
    
def print_length_distribution(dialogs):
    lengths = [len(dialog.split()) for dialog in dialogs]
    counter = Counter(lengths)
    
    print("\ndialog Length Distribution (in words):")
    for length, count in sorted(counter.items()):
        print(f"Length {length}: {count} dialogs")
    
    average_length = sum(lengths) / len(lengths) if lengths else 0
    print(f"\nAverage length: {average_length:.2f} words")
    
def print_average_word_and_char_count(dialogs):
    total_words = sum(len(dialog.split()) for dialog in dialogs)
    total_chars = sum(len(dialog) for dialog in dialogs)
    
    print(f"\nTotal number of dialogs: {len(dialogs)}")
    print(f"Average word count per dialog: {total_words / len(dialogs):.2f}")
    print(f"Average character count per dialog: {total_chars / len(dialogs):.2f}")

def print_sample_dialogs(dialogs, sample_size=5):
    print("\nRandom sample dialogs:\n")
    for dialog in random.sample(dialogs, sample_size):
        print(f"- {dialog}\n")

def print_dialogs_outliers(dialogs, count=2):
    sorted_dialogs = sorted(dialogs, key=lambda dialog: len(dialog.split()))

    print(f"\nShortest {count} dialogs:")
    for dialog in sorted_dialogs[:count]:
        print(f"- {dialog}\n")

    print(f"Longest {count} dialogs:")
    for dialog in sorted_dialogs[-count:]:
        print(f"- {dialog}\n")

def main():
    dialogs = load_dialogs()
    
    if dialogs:
        print_basic_statistics(dialogs)
        print_length_distribution(dialogs)
        print_average_word_and_char_count(dialogs)
        print_sample_dialogs(dialogs, 5)
        print_dialogs_outliers(dialogs, 2)
    else:
        print("No dialogs available for analysis.")

if __name__ == "__main__":
    main()