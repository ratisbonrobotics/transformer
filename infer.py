import torch
from model import LanguageModel
from tokenizer import encode_with_byte_fallback_utf8, decode_with_byte_fallback_utf8, load_vocab_from_json, VOCAB_SIZE

def generate_text(model, tokenizer, start_sequence, max_length=100, temperature=1.0):
    model.eval()
    tokens = encode_with_byte_fallback_utf8([start_sequence], tokenizer)[0]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to("cuda")

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_tensor)
            logits = outputs[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            input_tensor = torch.cat((input_tensor, next_token), dim=1)

    generated_text = decode_with_byte_fallback_utf8([input_tensor.squeeze().tolist()], tokenizer)
    return generated_text

# Load the trained model checkpoint
checkpoint = torch.load('checkpoint_1_114688.pth')
model = LanguageModel(VOCAB_SIZE).to("cuda")
model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#scaler.load_state_dict(checkpoint['gradscaler_state_dict'])

# Load the tokenizer
tokenizer = load_vocab_from_json("tokenizer.json")

# Set the starting sequence and generate text
start_sequence = "[USER] " + "Hello, how are you?".lower()
generated_text = generate_text(model, tokenizer, start_sequence)

print(generated_text[0])