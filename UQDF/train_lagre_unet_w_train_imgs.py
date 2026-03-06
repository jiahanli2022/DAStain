import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import shutil
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from autoencoder_models import LargeAutoEncoderUNet
from itertools import chain


class ImageDataset(Dataset):
    def __init__(self, root_dir, length=None):
        super().__init__()
        self.length = length
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

        self.image_files = []
        valid_ext = (".jpg","jpeg","png")

        for dirpath, _, filenames in os.walk(root_dir):  
            for f in filenames:
                if f.lower().endswith(valid_ext):
                    self.image_files.append(os.path.join(dirpath, f))

    def __len__(self):
        if self.length:
            return self.length
        else:
            return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path


if __name__ == '__main__':

    train_dir = "/mnt/nvme1n1/01.code/14.DAStain/TCGA-Dataset/TCGA-BRCA_Diagnostic-Splits/trainB/"
    train_dataset = ImageDataset(train_dir)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = LargeAutoEncoderUNet().to(device)
  
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    os.makedirs("Train_LargeUNet_w_TrainImgs/TCGA-BRCA_Diagnostic", exist_ok=True)

    epochs = 10
    print("Training...")
    for epoch in range(epochs):
        model.train()

        total_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            outputs = model(imgs)


            loss = criterion(outputs, imgs)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss/len(train_loader.dataset)
        print(f"[{epoch+1}/{epochs}] Loss = {avg_loss:.6f}")

        if (epoch + 1) % 1 == 0:
            model_path = f"Train_LargeUNet_w_TrainImgs/TCGA-BRCA_Diagnostic/{epoch+1:0>4d}_train_loss={avg_loss}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"saving at：{model_path}")
