import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms

# 1. Custom Dataset
class TopologyDataset(Dataset):
    def __init__(self, input_images, output_images, transform=None):
        self.input_images = input_images  # Shape: (N, H, W, 4)
        self.output_images = output_images  # Shape: (N, H, W, 1)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_img = self.input_images[idx]  # (H, W, 4)
        output_img = self.output_images[idx]  # (H, W, 1)

        # Convert to PyTorch tensors and adjust dimensions
        input_img = torch.tensor(input_img, dtype=torch.float32).permute(2, 0, 1)  # (4, H, W)
        output_img = torch.tensor(output_img, dtype=torch.float32).permute(2, 0, 1)  # (1, H, W)

        if self.transform:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)

        return input_img, output_img

# 2. U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = conv_block(256, 128)
        self.upconv5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec5 = conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Decoder with skip connections
        d4 = self.upconv4(e3)
        d4 = torch.cat([d4, e2], dim=1)  # Skip connection
        d4 = self.dec4(d4)
        d5 = self.upconv5(d4)
        d5 = torch.cat([d5, e1], dim=1)  # Skip connection
        d5 = self.dec5(d5)
        out = self.out_conv(d5)
        out = self.sigmoid(out)
        return out

# 3. Data Loading
def load_data(data_dir, img_size=(128, 128)):
    """
    Load dataset of input images (4 channels) and output topology images (1 channel).
    Assumes data_dir has subfolders 'inputs' and 'outputs' with .npy files.
    """
    input_images = []
    output_images = []

    input_dir = os.path.join(data_dir, 'inputs')
    output_dir = os.path.join(data_dir, 'outputs')

    for filename in os.listdir(input_dir):
        if filename.endswith('.npy'):
            input_img = np.load(os.path.join(input_dir, filename))
            output_img = np.load(os.path.join(output_dir, filename))

            # Resize if necessary (using torchvision)
            transform = transforms.Resize(img_size)
            input_img = transform(torch.tensor(input_img).permute(2, 0, 1)).permute(1, 2, 0).numpy()
            output_img = transform(torch.tensor(output_img).permute(2, 0, 1)).permute(1, 2, 0).numpy()

            input_images.append(input_img)  # (H, W, 4)
            output_images.append(output_img)  # (H, W, 1)

    return np.array(input_images), np.array(output_images)

# 4. Training Loop
def train_model(model, train_loader, val_loader, device, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save best model
        if epoch == 0 or val_loss < min(val_losses[:-1]):
            torch.save(model.state_dict(), 'best_model.pth')

    return train_losses, val_losses

# 5. Visualization
def plot_results(train_losses, val_losses, model, test_loader, device, num_samples=3):
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot sample predictions
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(test_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)

        for i in range(min(num_samples, inputs.size(0))):
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title('Input (Channel 1: Boundary)')
            plt.imshow(inputs[i, 0].cpu().numpy(), cmap='gray')
            plt.subplot(1, 3, 2)
            plt.title('Ground Truth Topology')
            plt.imshow(targets[i, 0].cpu().numpy(), cmap='gray')
            plt.subplot(1, 3, 3)
            plt.title('Predicted Topology')
            plt.imshow(predictions[i, 0].cpu().numpy(), cmap='gray')
            plt.show()

# 6. Main Execution
def main():
    # Parameters
    data_dir = 'path_to_your_dataset'  # Update with your dataset path
    img_size = (128, 128)
    batch_size = 16
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    X, y = load_data(data_dir, img_size)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create datasets and loaders
    train_dataset = TopologyDataset(X_train, y_train)
    val_dataset = TopologyDataset(X_val, y_val)
    test_dataset = TopologyDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    model = UNet(in_channels=4, out_channels=1).to(device)
    print(model)

    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, device, epochs)

    # Visualize results
    plot_results(train_losses, val_losses, model, test_loader, device)

if __name__ == '__main__':
    main()