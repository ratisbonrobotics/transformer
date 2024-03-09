import os
os.environ['PJRT_DEVICE'] = 'CPU'
import torch
from tqdm import tqdm
import torch.nn.functional as F
from model import Transformer, ModelArgs
import torch_xla.core.xla_model as xm
from tokenizer import encode_with_byte_fallback_utf8, load_vocab_from_json, decode_with_byte_fallback_utf8

# Constants
TOKENIZER_PATH = "tokenizer.json"
VOCAB_SIZE = 16384
CHECKPOINT_DIR = "checkpoints"

# Hyperparameters
SEQ_LENGTH = 1024
EMBEDDING_DIM = 768
NUM_HEADS = 8
DEPTH = 16
FEEDFORWARD_DIM = 2048

loaded_vocab = load_vocab_from_json(TOKENIZER_PATH)

# Load Model
model = Transformer(ModelArgs(EMBEDDING_DIM, DEPTH, EMBEDDING_DIM // NUM_HEADS, FEEDFORWARD_DIM, NUM_HEADS, 2, SEQ_LENGTH // 2, 1e-6, VOCAB_SIZE))

# Load Checkpoint
latest_checkpoint = max([os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR)], key=os.path.getctime)
checkpoint = torch.load(latest_checkpoint, map_location="cpu")
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Text Generation Function
def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_keep = sorted_indices[~sorted_indices_to_remove]
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.index_put_(tuple([indices_to_keep]), logits.index_select(0, indices_to_keep))
    return filtered_logits


def generate_text(start_text, max_length=50, temperature=1.0, top_p=0.9):
    model.eval()
    positions = torch.arange(0, SEQ_LENGTH)
    with torch.no_grad():
        tokens = encode_with_byte_fallback_utf8([start_text], loaded_vocab)[0]
        tokens = tokens[-SEQ_LENGTH:]
        for _ in tqdm(range(max_length), desc="Generating Text"):
            input_ids = torch.tensor([tokens], dtype=torch.long)
            logits = model(input_ids, positions)[0, -1, :]
            logits = top_p_sampling(logits, top_p=top_p, temperature=temperature)
            probabilities = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, 1).item()
            tokens.append(next_token_id)
            tokens = tokens[1:]
        generated_text = decode_with_byte_fallback_utf8([tokens], loaded_vocab)[0]
        return generated_text

start_text = "[USER] hi there, i've been hearing a lot about behavioral economics lately and i'm curious to learn more about how psychological, cognitive, emotional, and cultural factors affect economic decision-making and market outcomes. [ASSISTANT] that's a great topic to explore! behavioral economics is a fascinating field that blends insights from psychology and economics to understand why people make the decisions they do when it comes to money and markets. so, what specifically are you interested in learning more about? [USER] i'm particularly interested in how emotions play a role in our economic decisions. i've noticed that sometimes i make impulsive purchases when i'm feeling stressed or anxious, but other times i'm able to hold back and make more rational choices. why is that? [ASSISTANT] ah, emotions definitely play a significant role in decision-making. in the field of behavioral economics, researchers have found that emotions can influence our risk perception, time preferences, and even our willingness to trust others in economic transactions. when you're feeling stressed or anxious, for example, your brain may prioritize short-term rewards over long-term gains, leading to impulsive decisions. [USER] that makes a lot of sense. i've also heard about cognitive biases and how they can impact our economic decisions. can you explain how these biases come into play? [ASSISTANT] absolutely. cognitive biases are systematic errors in thinking that can influence our judgments and decisions. for example, confirmation bias is when we seek out information that confirms our preexisting beliefs, while anchoring bias is when we rely too heavily on the first piece of information we receive. these biases can lead us to make suboptimal choices in the realm of economics. [USER] it's incredible how our brains can trick us into making less-than-ideal decisions. i'm also curious about the role of cultural factors in economic decision-making. how does culture influence our choices in the market? [ASSISTANT] culture plays a significant role in shaping our values, beliefs, and attitudes towards money and markets. for example, different cultures may place varying levels of importance on saving versus spending, or have different norms around risk-taking. these cultural factors can influence how we approach economic decisions and interact with others in the marketplace. [USER] that's fascinating. it's clear that there are so many complex factors at play when it comes to economic decision-making. i'm excited to dive deeper into the world of behavioral economics and learn more about how our psychology affects our choices in the market. [ASSISTANT] i'm glad to hear that you're interested in exploring this topic further. behavioral economics offers valuable insights into the intricacies of human decision-making and can help us better understand the complexities of the economic world. let me know if you have any more questions or if there's a specific aspect of behavioral economics you'd like to delve into next! [USER] hi there! i'm interested in learning more about electrical engineering: circuits and systems. can you help me understand the principles of electrical circuits, electronics, electromagnetism, and signal processing? [ASSISTANT] of course! electrical engineering: circuits and systems is a fascinating field that covers a wide range of topics. let's start with electrical circuits.  [USER] great! i've heard that electrical circuits are the building blocks of all electronic devices. can you explain how they work? [ASSISTANT] absolutely. electrical circuits are paths through which electric current can flow. they are made up of components such as resistors, capacitors, inductors, and transistors, which control the flow of current. these components can be connected in different configurations, such as series or parallel, to create different circuit designs. [USER] that makes sense. is there a specific way to analyze and design electrical circuits? [ASSISTANT] yes, there are techniques such as kirchhoff's laws and ohm's law that can be used to analyze and design electrical circuits. kirchhoff's laws involve the conservation of charge and energy in a circuit, while ohm's law relates voltage, current, and resistance in a circuit. by applying these principles, engineers can predict how a circuit will behave and design circuits to meet specific requirements. [USER] that's interesting. what about electronics? how does it relate to electrical circuits? [ASSISTANT] electronics is the study of the behavior and design of electronic devices, which are composed of electrical circuits. electronics focuses on components such as diodes, transistors, and integrated circuits, which control the flow of electricity in a circuit. by combining these components in different ways, engineers can create a wide range of electronic devices, from smartphones to computers. [USER] i see. how does electromagnetism come into play in electrical engineering: circuits and systems? [ASSISTANT] electromagnetism is the study of the interaction between electric currents and magnetic fields. in electrical engineering, electromagnetism is used to design devices such as motors and generators, which convert electrical energy into mechanical energy and vice versa. understanding electromagnetism is crucial for designing efficient and reliable electrical systems. [USER] that's fascinating. and what about signal processing? how does it fit into the field of electrical engineering: circuits and systems? [ASSISTANT] signal processing is the study of manipulating signals, such as sound, images, and data, using electrical circuits and systems. it plays a crucial role in various applications, such as communication systems, medical imaging, and audio processing. engineers use signal processing techniques to analyze, filter, and enhance signals, allowing for efficient and accurate information processing. [USER] wow, that's a lot to cover! thank you for explaining these concepts to me. i'm excited to learn more about electrical engineering: circuits and systems. [ASSISTANT] you're welcome! if you have any more questions or need further clarification, feel free to ask. i'm here to help you navigate the fascinating world of electrical engineering. [USER] tell me about cosmology."
generated_text = generate_text(start_text, max_length=32, temperature=1.0, top_p=0.9)
print("Generated Text:\n", generated_text)