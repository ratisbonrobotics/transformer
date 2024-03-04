import os
import torch
import wandb
import pickle
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset
import torch_xla.core.xla_model as xm
from model import Transformer, ModelArgs
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from tokenizer import encode_with_byte_fallback_utf8, load_vocab_from_json, VOCAB_SIZE

os.environ['PJRT_DEVICE'] = 'TPU'

# Constants
NUM_EPOCHS = 128
TOKENIZER_PATH = "tokenizer.json"
DATASET_PATH = "dialogs.pkl"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
SEQ_LENGTH = 1024
EMBEDDING_DIM = 768
NUM_HEADS = 8
DEPTH = 16
FEEDFORWARD_DIM = 2048

class TextDataset(Dataset):
    def __init__(self, file_path, sequence_length, loaded_vocab):
        self.sequence_length = sequence_length

        with open(file_path, "rb") as file:
            dialogs = pickle.load(file)

        self.dialogs = encode_with_byte_fallback_utf8(dialogs, loaded_vocab)
        self.dialogs = [dialog for dialog in self.dialogs]

    def __len__(self):
        return len(self.dialogs) - (self.sequence_length + 1)
    
    def __getitem__(self, idx):
        input = torch.tensor(self.texts[idx : idx + self.sequence_length], dtype=torch.long)
        output = torch.tensor(self.texts[idx + 1: idx + self.sequence_length + 1], dtype=torch.long)
        return input, output
    

def _train_model(rank):
    device = xm.xla_device()
    checkpoint_path = None

    loaded_vocab = load_vocab_from_json(TOKENIZER_PATH)
    trainset = TextDataset(DATASET_PATH, SEQ_LENGTH, loaded_vocab) 
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=xm.xrt_world_size(), rank=rank, shuffle=True, drop_last=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=train_sampler, num_workers=8)

    model = Transformer(ModelArgs(EMBEDDING_DIM, DEPTH, EMBEDDING_DIM // NUM_HEADS, FEEDFORWARD_DIM, NUM_HEADS, 2, SEQ_LENGTH // 2, 1e-6, VOCAB_SIZE)).to(device)
    
    if rank == 0:
        wandb.init(project="transformer")
        print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    positions = torch.arange(0, SEQ_LENGTH).to(device)
    model.train()
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)

        parallel_loader = pl.ParallelLoader(trainloader, [device])
        train_loader = parallel_loader.per_device_loader(device)

        with tqdm(enumerate(train_loader), total=len(train_loader), disable=(rank != 0)) as pbar:
            for batch_idx, data in pbar:
                image, inputs, labels = data[0], data[1], data[2]
                optimizer.zero_grad()
                outputs = model(image, inputs, positions)
                loss = criterion(outputs.transpose(1, 2), labels)
                loss.backward()
                xm.optimizer_step(optimizer, barrier=True)

                pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training Loss: {loss.item():.4f}")

                if rank == 0:
                    wandb.log({"loss": loss.item()})
                    if batch_idx > 0 and batch_idx % 512 == 0:
                        checkpoint_path_new = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
                        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint_path_new)
                        if checkpoint_path: os.remove(checkpoint_path)
                        checkpoint_path = checkpoint_path_new

if __name__ == '__main__':
    xmp.spawn(_train_model)