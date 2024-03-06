import tqdm
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Define the model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

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
        self.scale = head_dim**-0.5
        self.wq = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, self.n_heads, seqlen, -1)
        xk = xk.view(bsz, self.n_heads, seqlen, -1)
        xv = xv.view(bsz, self.n_heads, seqlen, -1)

        scores = torch.einsum("bhid,bhjd->bhij", xq, xk) * self.scale
        scores = nn.functional.softmax(scores, dim=-1)

        output = torch.einsum("bhij,bhjd->bhid", scores, xv)
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
    def __init__(self, num_blocks=2, num_heads=4, hidden_dim=32, ff_dim=128):
        super(MNISTModel, self).__init__()

        self.linear_in = nn.Linear(7 * 7, hidden_dim, bias=False)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(num_heads, hidden_dim, ff_dim) for _ in range(num_blocks)
        ])

        self.linear_out = nn.Linear(hidden_dim * 16, 10, bias=False)

    def forward(self, x: torch.Tensor):
        # Split the image into 16 patches (4x4 grid)
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(x.size(0), 16, -1)

        # Linear transformation of patches
        patches = self.linear_in(patches)
        patches = self.pos_encoding(patches)

        for block in self.transformer_blocks:
            patches = block(patches)

        x = patches.view(patches.size(0), -1)

        x = self.linear_out(x)
        return x

# Load the MNIST dataset
train_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='./data', train=False, transform=ToTensor())

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True, num_workers=4)

# Create the model
model = MNISTModel()
model = torch.jit.script(model)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for images, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}"):
        images, labels = images, labels
        
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
            images, labels = images, labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.2f}%")