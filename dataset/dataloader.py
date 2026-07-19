import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class DIV2KDataset(Dataset):
    """
    PyTorch Dataset for DIV2K Image Super-Resolution pairs.
    Loads Low-Resolution (LR) images (128x128) and High-Resolution (HR) ground truth (512x512).
    Applies data augmentations (flips, rotations) during training mode using PIL.
    """
    def __init__(self, root_dir, split="train", hr_size=512, lr_size=128, is_train=True):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.is_train = is_train

        self.hr_dir = os.path.join(root_dir, split, "hr")
        self.lr_dir = os.path.join(root_dir, split, "lr")

        self.hr_files = sorted(glob.glob(os.path.join(self.hr_dir, "*.png")) + glob.glob(os.path.join(self.hr_dir, "*.jpg")))
        self.lr_files = sorted(glob.glob(os.path.join(self.lr_dir, "*.png")) + glob.glob(os.path.join(self.lr_dir, "*.jpg")))

        if len(self.hr_files) == 0:
            raise FileNotFoundError(f"No HR images found in '{self.hr_dir}'. Please run 'python dataset/download_div2k.py' first.")

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = self.lr_files[idx] if idx < len(self.lr_files) else hr_path

        hr_img = Image.open(hr_path).convert("RGB")
        
        if os.path.exists(lr_path) and idx < len(self.lr_files):
            lr_img = Image.open(lr_path).convert("RGB")
        else:
            lr_img = hr_img.resize((self.lr_size, self.lr_size), Image.BICUBIC)

        if hr_img.size != (self.hr_size, self.hr_size):
            hr_img = hr_img.resize((self.hr_size, self.hr_size), Image.BICUBIC)
        if lr_img.size != (self.lr_size, self.lr_size):
            lr_img = lr_img.resize((self.lr_size, self.lr_size), Image.BICUBIC)

        # Data Augmentation (Train mode only)
        if self.is_train:
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
                lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
                lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)

            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                angle = rot_k * 90
                hr_img = hr_img.rotate(angle)
                lr_img = lr_img.rotate(angle)

        # Convert PIL to PyTorch Tensor [C, H, W] range [0, 1]
        hr_np = np.array(hr_img).transpose(2, 0, 1).astype(np.float32) / 255.0
        lr_np = np.array(lr_img).transpose(2, 0, 1).astype(np.float32) / 255.0

        hr_tensor = torch.from_numpy(hr_np)
        lr_tensor = torch.from_numpy(lr_np)

        return {"lr": lr_tensor, "hr": hr_tensor, "filename": os.path.basename(hr_path)}


def get_dataloaders(dataset_dir="dataset", batch_size=4, num_workers=0, hr_size=512, lr_size=128):
    train_dataset = DIV2KDataset(dataset_dir, split="train", hr_size=hr_size, lr_size=lr_size, is_train=True)
    valid_dataset = DIV2KDataset(dataset_dir, split="valid", hr_size=hr_size, lr_size=lr_size, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader
