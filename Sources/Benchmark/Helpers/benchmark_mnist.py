import torch
import torch.nn as nn
import torch.optim as optim
import time

# 1. Architecture definition — identical to the C# one
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size)
    def forward(self, x):
        # Mirrors the logic from ResidualBlock.cs
        return torch.relu(self.linear(x)) + x

class MNISTResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3) 
        self.bn1 = nn.BatchNorm1d(1352)             
        self.res1 = ResidualBlock(1352)             
        self.fcOut = nn.Linear(8, 10)               

    def forward(self, x):
        x = torch.relu(self.conv1(x))               
        x = torch.max_pool2d(x, 2)                  
        x = x.view(x.size(0), -1)                   
        x = self.bn1(x)                             
        x = self.res1(x)                            
        x = x.view(-1, 8, 13, 13)                   
        x = torch.mean(x, dim=(2, 3)) # Global Average Pool
        return self.fcOut(x)                        

# 2. Device configuration (force CPU for a fair comparison with the C# engine)
device = torch.device("cpu")
model = MNISTResNet().to(device)

# AdamW parameters matching the Adam class
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

# Synthetic data: 60,000 samples, batch 64 => 937 batches
dummy_x = torch.randn(64, 1, 28, 28).to(device)
dummy_y = torch.randint(0, 10, (64,)).to(device)

print(f"--- START BENCHMARK: PyTorch on CPU ---")
total_start = time.time()

for epoch in range(5):
    epoch_start = time.time()
    model.train()
    
    for _ in range(937):
        optimizer.zero_grad()
        output = model(dummy_x)
        loss = criterion(output, dummy_y)
        loss.backward()
        optimizer.step()
    
    epoch_end = time.time()
    print(f"> EPOCH {epoch+1} DONE | Time: {(epoch_end - epoch_start)*1000:.0f}ms")

total_end = time.time()
print(f"---------------------------------------")
print(f"Benchmark finished! Total time: {total_end - total_start:.2f}s")