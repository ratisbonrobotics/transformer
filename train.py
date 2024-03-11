import os
import tqdm
import wandb
import torch
import pickle
from model import LanguageModel
from tokenizer import encode_with_byte_fallback_utf8, load_vocab_from_json, VOCAB_SIZE

# screen -L -S train -t train bash -c 'cd /root/transformer && /opt/conda/bin/python /root/transformer/train.py'

# Constants
NUM_EPOCHS = 128
SEQ_LENGTH = 2048
WANDB = True
WARMUP_STEPS = 1000
TARGET_LR = 1e-4

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, sequence_length, loaded_vocab, cache_file="dialogs_cache.pkl"):
        self.sequence_length = sequence_length

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as file:
                self.dialogs = pickle.load(file)
        else:
            with open(file_path, "rb") as file:
                loaded_dialogs = pickle.load(file)
            self.dialogs = encode_with_byte_fallback_utf8(loaded_dialogs, loaded_vocab)
            self.dialogs = [item for sublist in self.dialogs for item in sublist]
            with open(cache_file, "wb") as file:
                pickle.dump(self.dialogs, file)

    def __len__(self):
        return (len(self.dialogs) - (self.sequence_length + 1)) // self.sequence_length
    
    def __getitem__(self, idx):
        idx = idx * self.sequence_length
        inputs = torch.tensor(self.dialogs[idx : idx + self.sequence_length], dtype=torch.long)
        labels = torch.tensor(self.dialogs[idx + 1: idx + self.sequence_length + 1], dtype=torch.long)
        return inputs, labels

# Create Dataset and Dataloader
train_dataset = TextDataset("open_orca.pkl", SEQ_LENGTH, load_vocab_from_json("tokenizer.json"), cache_file="open_orca_cache.pkl")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=8)

# Create the model
model = torch.compile(LanguageModel(VOCAB_SIZE, SEQ_LENGTH).to("cuda"))
print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
if WANDB: wandb.init(project="primitive")

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scaler = torch.cuda.amp.GradScaler()

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    with tqdm.tqdm(train_loader) as pbar:
        for batch, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to("cuda"), labels.to("cuda")

            # Set the learning rate for the optimizer based on the warmup schedule
            for param_group in optimizer.param_groups:
                param_group['lr'] = min(TARGET_LR * ((epoch * len(train_loader) + batch) / WARMUP_STEPS), TARGET_LR)
            
            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs.transpose(1, 2), labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log progress
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Loss: {loss.item():.4f}")
            if WANDB: wandb.log({"loss": loss.item()})

            # Periodically save checkpoint
            if (batch + 1) % 512 == 0:
                for f in os.listdir('.'):
                    if f.startswith('checkpoint_') and f.endswith('.pth'):
                        os.remove(f)
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'gradscaler_state_dict': scaler.state_dict()}, f'checkpoint_{epoch+1}_{batch+1}.pth')