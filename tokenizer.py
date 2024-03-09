import re
import os
import json
import tqdm
import pickle
from collections import Counter

TOKENIZE_REGEX = r"(\[USER\]|\[ASSISTANT\]|\b[\w.]+@[\w.]+\.\w+\b|\w+|\d+|[^\w\s])"
VOCAB_SIZE = 32768

def load_textlist(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return []
    with open(file_path, "rb") as file:
        dialogs = pickle.load(file)
    return dialogs

def tokenize(dialogs, vocab_size):
    special_tokens = {"[USER]", "[ASSISTANT]"}
    word_list = [word for dialog in tqdm.tqdm(dialogs) for word in re.findall(TOKENIZE_REGEX, dialog)]
    word_count = Counter(word_list)
    vocabulary = {token: idx for idx, token in enumerate(special_tokens, start=256)}
    for token in special_tokens:
        word_count.pop(token, None)
    remaining_vocab_size = vocab_size - len(special_tokens)
    most_common_words = [(word, count) for word, count in word_count.most_common(remaining_vocab_size) if len(word) > 1 or word in special_tokens]
    for i, (word, _) in enumerate(most_common_words, start=len(vocabulary) + 256):
        vocabulary[word] = i
    return vocabulary

def save_vocab_to_json(vocabulary, output_file):
    """Save the vocabulary to a json file."""
    with open(output_file, 'w') as json_file:
        json.dump(vocabulary, json_file, indent=4)
    print(f"Vocabulary saved to {output_file}")

def load_vocab_from_json(input_file):
    """Load the vocabulary from a json file."""
    with open(input_file, 'r') as json_file:
        vocabulary = json.load(json_file)
    return vocabulary

def encode_with_byte_fallback_utf8(dialogs, vocabulary):
    if isinstance(dialogs, str):
        dialogs = [dialogs]
    encoded_dialogs = []
    for dialog in dialogs:
        encoded_dialog = []
        for word in re.findall(TOKENIZE_REGEX, dialog):
            if word in vocabulary:
                encoded_dialog.append(vocabulary[word])
            else:
                encoded_dialog.extend(word.encode('utf-8'))
        encoded_dialogs.append(encoded_dialog)
    return encoded_dialogs

def decode_with_byte_fallback_utf8(token_lists, vocabulary):
    inverted_vocabulary = {v: k for k, v in vocabulary.items()}
    decoded_dialogs = []
    for token_list in token_lists:
        decoded_dialog = []
        byte_sequence = []
        for token in token_list:
            if token in inverted_vocabulary:
                if byte_sequence:
                    decoded_bytes = ' '.join(chr(b) for b in byte_sequence)
                    decoded_dialog.append(decoded_bytes)
                    byte_sequence = []
                decoded_dialog.append(inverted_vocabulary[token])
            else:
                byte_sequence.append(token)
        if byte_sequence:
            decoded_bytes = ' '.join(chr(b) for b in byte_sequence)
            decoded_dialog.append(decoded_bytes)
        decoded_dialogs.append(' '.join(decoded_dialog))
    return decoded_dialogs

if __name__ == "__main__":
    # Training
    dialogs = load_textlist("open_orca.pkl")

    vocab_size = VOCAB_SIZE
    vocab_file = "tokenizer.json"
    
    vocabulary = tokenize(dialogs, vocab_size)
    save_vocab_to_json(vocabulary, vocab_file)
    
    # Testing
    dialogs_val = load_textlist("open_orca.pkl")
    loaded_vocab = load_vocab_from_json(vocab_file)
    
    encoded_dialogs = encode_with_byte_fallback_utf8([dialogs_val[1]], loaded_vocab)
    print("Encoded with smart tokenization and byte fallback:", encoded_dialogs[0])
    
    decoded_dialogs = decode_with_byte_fallback_utf8(encoded_dialogs, loaded_vocab)
    print("Decoded with smart tokenization and byte fallback:", decoded_dialogs[0])