import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# -------------------------
# Dataset
# -------------------------
class LBracketDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        """
        input_dir: contains input images (BC + Fx + Fy)
        target_dir: contains target density maps
        """
        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        x = np.load(self.input_files[idx])   # shape: (3,H,W) -> BC + Fx + Fy
        y = np.load(self.target_files[idx])  # shape: (1,H,W) -> density
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

# -------------------------
# UNet model (good for images)
# -------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32,64,128]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part
        for feature in features:
            self.downs.append(self.conv_block(in_channels, feature))
            in_channels = feature
            
        # Up part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self.conv_block(feature*2, feature))
        
        self.bottleneck = self.conv_block(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
        
        return torch.sigmoid(self.final_conv(x))
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

# -------------------------
# Training setup
# -------------------------
def train_model(train_loader, val_loader, device, epochs=50, lr=1e-3):
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # predicting density map
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        
        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                val_loss += criterion(y_pred, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
    
    print("Training finished. Best validation loss:", best_val_loss)
    return model

# -------------------------
# Usage example
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = LBracketDataset("train_inputs/", "train_targets/")
    val_dataset = LBracketDataset("val_inputs/", "val_targets/")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = train_model(train_loader, val_loader, device, epochs=50, lr=1e-3)
