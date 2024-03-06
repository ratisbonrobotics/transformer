import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Define the model
import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self, num_heads=8, hidden_dim=256):
        super(MNISTModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
       
        self.linear_in = nn.Linear(14 * 14, hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.linear_out = nn.Linear(hidden_dim, 10)
        self.silu = nn.SiLU()
       
    def forward(self, x: torch.Tensor):
        # Split the image into four patches
        patches = x.unfold(2, 14, 14).unfold(3, 14, 14)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(x.size(0), 4, -1)
       
        # Linear transformation of patches
        patches = self.linear_in(patches)
       
        # Perform self-attention on patches
        attn_output, _ = self.self_attn(patches, patches, patches)
       
        # Average the patch representations
        x = attn_output.mean(dim=1)
       
        x = self.silu(x)
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