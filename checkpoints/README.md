# Model Checkpoints Directory

This directory stores trained model weights for the **ESRGAN Super-Resolution Generator** (`RRDBNet`) and **VGG Discriminator**.

## Pretrained Weights & Auto-Initialization

- Model weights (`esrgan_best.pth`, `esrgan_latest.pth`) are dynamically managed at runtime.
- When running `app.py` or `train.py`, the system automatically initializes architecture parameters (`num_features`, `num_blocks`) and loads existing weights from this directory if available.
- Large binary weights (`*.pth`, `*.pt`, `*.ckpt`) are excluded from Git version control to keep the repository lightweight and fast.

## Training Custom Checkpoints

To train your own checkpoints on custom images or the DIV2K dataset:
```bash
python train.py --epochs 10 --batch_size 4
```
Trained checkpoints will automatically save to `checkpoints/esrgan_best.pth`.
