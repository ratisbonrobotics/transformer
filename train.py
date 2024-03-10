import os
import tqdm
import wandb
import torch
import pickle
import functools
from model import LanguageModel
from tokenizer import encode_with_byte_fallback_utf8, load_vocab_from_json, VOCAB_SIZE
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 train.py

torch.distributed.init_process_group(backend='nccl')

# Constants
NUM_EPOCHS = 128
SEQ_LENGTH = 2048
WANDB = False

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
        return len(self.dialogs) - (self.sequence_length + 1)
    
    def __getitem__(self, idx):
        inputs = torch.tensor(self.dialogs[idx : idx + self.sequence_length], dtype=torch.long)
        labels = torch.tensor(self.dialogs[idx + 1: idx + self.sequence_length + 1], dtype=torch.long)
        return inputs, labels

# Create Dataset and Dataloader
train_dataset = TextDataset("open_orca.pkl", SEQ_LENGTH, load_vocab_from_json("tokenizer.json"), cache_file="open_orca_cache.pkl")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=8)

# Create the model
model = FSDP(LanguageModel(VOCAB_SIZE).to(f"cuda:{os.environ['RANK']}"), mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16), auto_wrap_policy=functools.partial(size_based_auto_wrap_policy, min_num_params=20000))
print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
if WANDB: wandb.init(project="primitive")

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

# Potentially restore checkpoint
if os.path.exists("checkpoint_1_512.pth"):
    checkpoint = torch.load('checkpoint_1_512.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    with tqdm.tqdm(train_loader) as pbar:
        for batch, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(f"cuda:{os.environ['RANK']}"), labels.to(f"cuda:{os.environ['RANK']}")
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log progress
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Loss: {loss.item():.4f}")
            if WANDB: wandb.log({"loss": loss.item()})

            # Periodically save checkpoint
            if (batch + 1) % 512 == 0 and torch.distributed.get_rank() == 0:
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint_{epoch+1}_{batch+1}.pth')
