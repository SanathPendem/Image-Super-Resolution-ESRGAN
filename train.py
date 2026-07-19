import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.download_div2k import generate_synthetic_div2k_samples
from dataset.dataloader import get_dataloaders
from models.generator import RRDBNet
from models.discriminator import VGGDiscriminator
from loss.perceptual_loss import ESRGANLoss
from utils.metrics import evaluate_batch

def train_esrgan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing ESRGAN Training on Device: {device}")

    if not os.path.exists(os.path.join(args.dataset_dir, "train", "hr")):
        print("[*] Training directory missing. Generating DIV2K sample dataset...")
        generate_synthetic_div2k_samples(base_dir=args.dataset_dir, num_train=10, num_valid=3)

    train_loader, valid_loader = get_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        hr_size=args.hr_size,
        lr_size=args.lr_size
    )

    generator = RRDBNet(num_features=64, num_blocks=args.num_blocks).to(device)
    discriminator = VGGDiscriminator(num_features=64, img_size=args.hr_size).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.999))

    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.5)

    criterion = ESRGANLoss(device=device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_psnr = 0.0

    print(f"[+] Generator Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"[+] Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]")
        for batch in pbar:
            lr_imgs = batch["lr"].to(device)
            hr_imgs = batch["hr"].to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            with torch.no_grad():
                sr_imgs = generator(lr_imgs)

            pred_real = discriminator(hr_imgs)
            pred_fake = discriminator(sr_imgs.detach())

            loss_D = criterion.compute_discriminator_loss(pred_real, pred_fake)
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            pred_real = discriminator(hr_imgs)
            pred_fake = discriminator(sr_imgs)

            loss_G, loss_dict = criterion.compute_generator_loss(sr_imgs, hr_imgs, pred_real, pred_fake)
            loss_G.backward()
            
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()

            pbar.set_postfix({
                "Loss_G": f"{loss_G.item():.4f}",
                "Loss_D": f"{loss_D.item():.4f}",
                "Pixel": f"{loss_dict['loss_pixel']:.4f}"
            })

        scheduler_G.step()
        scheduler_D.step()

        # ---------------------
        #  Validation Pipeline
        # ---------------------
        generator.eval()
        val_metrics = {"psnr": [], "ssim": [], "lpips": []}

        with torch.no_grad():
            for val_batch in valid_loader:
                v_lr = val_batch["lr"].to(device)
                v_hr = val_batch["hr"].to(device)
                v_sr = generator(v_lr)

                m = evaluate_batch(v_sr, v_hr, device=device)
                val_metrics["psnr"].append(m["psnr"])
                val_metrics["ssim"].append(m["ssim"])
                val_metrics["lpips"].append(m["lpips"])

        avg_psnr = float(np.mean(val_metrics["psnr"]))
        avg_ssim = float(np.mean(val_metrics["ssim"]))
        avg_lpips = float(np.mean(val_metrics["lpips"]))

        print(f" -> Epoch {epoch} Validation Metrics: PSNR={avg_psnr:.2f}dB | SSIM={avg_ssim:.4f} | LPIPS={avg_lpips:.4f}")

        latest_path = os.path.join(args.checkpoint_dir, "esrgan_latest.pth")
        torch.save({
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "metrics": {"psnr": avg_psnr, "ssim": avg_ssim, "lpips": avg_lpips}
        }, latest_path)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_path = os.path.join(args.checkpoint_dir, "esrgan_best.pth")
            torch.save(generator.state_dict(), best_path)
            print(f"    [*] Best model checkpoint saved to '{best_path}'")

    elapsed = time.time() - start_time
    print(f"[+] Training completed in {elapsed:.2f} seconds. Best PSNR: {best_psnr:.2f}dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESRGAN Training Script")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to save checkpoints")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of RRDB blocks for model")
    parser.add_argument("--lr_g", type=float, default=1e-4, help="Learning rate for Generator")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="Learning rate for Discriminator")
    parser.add_argument("--hr_size", type=int, default=512, help="High-resolution image size")
    parser.add_argument("--lr_size", type=int, default=128, help="Low-resolution image size")
    args = parser.parse_args()

    train_esrgan(args)
