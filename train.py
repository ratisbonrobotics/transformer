import os
import io
import torch
import wandb
import pickle
import random
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torch.optim as optim
from base64 import b64decode
from datetime import datetime
from torchvision import transforms
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

class TextImageDataset(Dataset):
    def __init__(self, file_path, sequence_length, loaded_vocab):
        self.sequence_length = sequence_length

        with open(file_path, "rb") as file:
            text_image_data = pickle.load(file)

        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.images = []
        self.texts = []

        for image, text in text_image_data.items():
            img = Image.open(io.BytesIO(b64decode(image)))
            img = transform(img)
            self.images.append(img)
            self.texts.append(encode_with_byte_fallback_utf8([text], loaded_vocab)[0])

    def __len__(self):
        return len(self.images) * max(len(text) for text in self.texts)
    
    def __getitem__(self, idx):
        index = idx % len(self.images)
        while True:
            try:
                start = random.randint(0, len(self.texts[index]) - self.sequence_length - 1)
                end = start + self.sequence_length
                inputs_tensor = torch.tensor(self.texts[index][start:end], dtype=torch.long)
                targets_tensor = torch.tensor(self.texts[index][start+1:end+1], dtype=torch.long)
                return self.images[index], inputs_tensor, targets_tensor
            except Exception as e:
                # print(f"Error fetching data: {e} - Choosing different image...")
                index = (index + 1) % len(self.images)

def _train_model(rank):
    device = xm.xla_device()
    checkpoint_path = None

    loaded_vocab = load_vocab_from_json(TOKENIZER_PATH)
    trainset = TextImageDataset(DATASET_PATH, SEQ_LENGTH, loaded_vocab) 
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