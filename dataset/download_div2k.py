import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_synthetic_div2k_samples(base_dir="dataset", num_train=20, num_valid=5, hr_size=512, lr_size=128):
    """
    Generates high-resolution (512x512) synthetic images with sharp geometric patterns, 
    gradients, and high-frequency textures, along with downsampled low-resolution (128x128) pairs using PIL.
    """
    train_hr_dir = os.path.join(base_dir, "train", "hr")
    train_lr_dir = os.path.join(base_dir, "train", "lr")
    valid_hr_dir = os.path.join(base_dir, "valid", "hr")
    valid_lr_dir = os.path.join(base_dir, "valid", "lr")

    for d in [train_hr_dir, train_lr_dir, valid_hr_dir, valid_lr_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"[*] Preparing DIV2K synthetic sample dataset in '{base_dir}'...")

    def create_rich_pattern(idx, size=512):
        np.random.seed(idx * 42)
        
        # 1. Background gradient
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)
        r = (np.sin(xx * 10 + idx) + 1) / 2 * 255
        g = (np.cos(yy * 12 + idx) + 1) / 2 * 255
        b = (np.sin((xx + yy) * 8 + idx) + 1) / 2 * 255
        
        img_arr = np.zeros((size, size, 3), dtype=np.uint8)
        img_arr[:, :, 0] = r.astype(np.uint8)
        img_arr[:, :, 1] = g.astype(np.uint8)
        img_arr[:, :, 2] = b.astype(np.uint8)

        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)

        # 2. Add sharp shapes & high frequency textures
        for _ in range(15):
            cx, cy = np.random.randint(50, size-50), np.random.randint(50, size-50)
            radius = np.random.randint(15, 80)
            color = tuple([int(c) for c in np.random.randint(0, 256, 3)])
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color, outline=(255, 255, 255), width=2)

        # 3. Grid pattern for high frequency detail
        for i in range(0, size, 32):
            draw.line([(i, 0), (i, size)], fill=(200, 200, 200), width=1)
            draw.line([(0, i), (size, i)], fill=(200, 200, 200), width=1)

        # 4. Text lines for super-res texture challenge
        draw.text((30, size - 40), f"DIV2K Sample #{idx+1:04d}", fill=(255, 255, 255))
        draw.text((30, 40), "ESRGAN Super-Resolution System", fill=(0, 255, 255))

        return img

    def save_paired_dataset(count, hr_dir, lr_dir, prefix="img"):
        for i in range(count):
            hr_img = create_rich_pattern(i, size=hr_size)
            filename = f"{prefix}_{i+1:04d}.png"
            hr_path = os.path.join(hr_dir, filename)
            lr_path = os.path.join(lr_dir, filename)

            # Save HR (512x512)
            hr_img.save(hr_path)

            # Generate LR (128x128) using Bicubic Downsampling
            lr_img = hr_img.resize((lr_size, lr_size), Image.BICUBIC)
            lr_img.save(lr_path)

    save_paired_dataset(num_train, train_hr_dir, train_lr_dir, prefix="train")
    save_paired_dataset(num_valid, valid_hr_dir, valid_lr_dir, prefix="valid")

    print(f"[+] Created {num_train} training pairs and {num_valid} validation pairs.")
    print(f"    HR dimension: {hr_size}x{hr_size} | LR dimension: {lr_size}x{lr_size}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DIV2K sample dataset for ESRGAN training")
    parser.add_argument("--num_train", type=int, default=20, help="Number of training samples")
    parser.add_argument("--num_valid", type=int, default=5, help="Number of validation samples")
    args = parser.parse_args()

    generate_synthetic_div2k_samples(num_train=args.num_train, num_valid=args.num_valid)
