# Dataset Pipeline Directory

This directory contains the dataset preparation scripts and PyTorch DataLoaders for 4× Super-Resolution training.

## Dataset Preparation & Downloading

To generate or download synthetic DIV2K paired low-resolution ($128\times128$) and high-resolution ($512\times512$) training samples out-of-the-box, run:

```bash
python dataset/download_div2k.py --num_train 30 --num_valid 10
```

## Data Loader Features

- `dataset/dataloader.py`: Implements PyTorch `Dataset` and `DataLoader` classes with real-time data augmentations including random horizontal flips, vertical flips, and random $90^\circ$ rotations.
- Paired dataset image folders (`dataset/train/`, `dataset/valid/`) are excluded from Git version control.
