import os
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

class TopologyDataset(Dataset):
    def __init__(self, input_folder, output_folder):
        self.input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.mat')])
        self.output_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.mat')])
        self.input_folder = input_folder
        self.output_folder = output_folder
        assert len(self.input_files) == len(self.output_files), "Input/output files mismatch"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load input
        X_mat = loadmat(os.path.join(self.input_folder, self.input_files[idx]))
        X = X_mat['X']  # H x W x C
        X = torch.tensor(X.transpose(2,0,1), dtype=torch.float32)  # C x H x W

        # Load output
        Y_mat = loadmat(os.path.join(self.output_folder, self.output_files[idx]))
        Y = Y_mat['Y']  # H x W
        Y = torch.tensor(Y[np.newaxis, :, :], dtype=torch.float32)    # 1 x H x W

        return X, Y
# Create DataLoader
from torch.utils.data import DataLoader

dataset = TopologyDataset('data/train/input/', 'data/train/output/')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# 
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Sigmoid()  # outputs in [0,1] for density
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleUNet().to(device)
criterion = nn.MSELoss()  # or BCELoss if density is binary
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 50

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for X_batch, Y_batch in dataloader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")


model.eval()
with torch.no_grad():
    # Suppose new_input is H x W x 3 numpy array
    new_input = torch.tensor(new_input.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)
    pred = model(new_input)  # shape: 1 x 1 x H x W
    pred = pred.squeeze().cpu().numpy()  # H x W
