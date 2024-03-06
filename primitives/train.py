import tqdm
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

# Define the model
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed

class FeedForward(nn.Module):
    def __init__(self, hidden_dim=32, ff_dim=128):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ff_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    def __init__(self, n_heads, hidden_dim, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.scale = head_dim ** -0.5
        self.q_linear = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.k_linear = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.v_linear = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_linear(x).view(bsz, seqlen, self.n_heads, -1)
        k = self.k_linear(x).view(bsz, seqlen, self.n_heads, -1)
        v = self.v_linear(x).view(bsz, seqlen, self.n_heads, -1)

        scores = torch.einsum("bsid,bsjd->bsij", q, k) * self.scale
        scores = nn.functional.softmax(scores, dim=-1)

        output = torch.einsum("bsij,bsjd->bsid", scores, v)
        output = output.view(bsz, seqlen, -1)

        return self.wo(output)

class TransformerBlock(nn.Module):
    def __init__(self, num_heads=4, hidden_dim=32, ff_dim=128):
        super().__init__()
        self.n_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention = Attention(num_heads, hidden_dim, hidden_dim // num_heads)
        self.feed_forward = FeedForward(hidden_dim, ff_dim)
        self.attention_norm = RMSNorm(hidden_dim)
        self.ffn_norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x))
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out

class MNISTModel(nn.Module):
    def __init__(self, num_blocks=4, num_heads=4, hidden_dim=512, ff_dim=1024):
        super(MNISTModel, self).__init__()
        self.pix_emb = nn.Embedding(256, hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(num_heads, hidden_dim, ff_dim) for _ in range(num_blocks)
        ])
        self.linear_out = nn.Linear(hidden_dim * 28 * 28, 10, bias=False)

    def forward(self, x: torch.Tensor):
        pixels = (x.view(x.size(0), -1) * 255).long()
        pixels_embedded = self.pix_emb(pixels)
        
        for block in self.transformer_blocks:
            pixels_embedded = block(pixels_embedded)

        x = pixels_embedded.view(pixels_embedded.size(0), -1)
        x = self.linear_out(x)
        return x

# Load the MNIST dataset
train_dataset = FashionMNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = FashionMNIST(root='./data', train=False, transform=ToTensor())

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=190, shuffle=True, drop_last=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=190, shuffle=False, drop_last=True, num_workers=4)

# Create the model
model = MNISTModel().to("cuda")
print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for images, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}"):
        images, labels = images.to("cuda"), labels.to("cuda")
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.2f}%")