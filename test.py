import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
import argparse
import glob
import numpy as np
from PIL import Image
import torch

from dataset.download_div2k import generate_synthetic_div2k_samples
from models.generator import RRDBNet, SRResNet
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips

def run_bicubic_np(lr_img_np, hr_size=512):
    lr_pil = Image.fromarray(lr_img_np)
    bicubic_pil = lr_pil.resize((hr_size, hr_size), Image.BICUBIC)
    return np.array(bicubic_pil)

def benchmark_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"[*] ESRGAN Super-Resolution Benchmark Suite (Device: {device})")
    print(f"============================================================")

    valid_hr_dir = os.path.join(args.dataset_dir, "valid", "hr")
    valid_lr_dir = os.path.join(args.dataset_dir, "valid", "lr")

    if not os.path.exists(valid_hr_dir) or len(glob.glob(os.path.join(valid_hr_dir, "*.png"))) == 0:
        print("[*] Generating sample validation images for benchmarking...")
        generate_synthetic_div2k_samples(base_dir=args.dataset_dir, num_train=10, num_valid=5)

    hr_files = sorted(glob.glob(os.path.join(valid_hr_dir, "*.png")) + glob.glob(os.path.join(valid_hr_dir, "*.jpg")))

    esrgan = RRDBNet(num_features=64, num_blocks=args.num_blocks).to(device)
    srgan = SRResNet(num_features=64, num_blocks=4).to(device)

    best_pth = os.path.join(args.checkpoint_dir, "esrgan_best.pth")
    if os.path.exists(best_pth):
        try:
            esrgan.load_state_dict(torch.load(best_pth, map_location=device), strict=False)
            print(f"[+] Loaded trained ESRGAN weights from '{best_pth}'")
        except Exception as e:
            print(f"[!] Info: Using initialized ESRGAN weights ({e})")
    else:
        print("[!] Info: No checkpoint found. Evaluating initialized models for baseline verification.")

    esrgan.eval()
    srgan.eval()

    results = {
        "Bicubic": {"psnr": [], "ssim": [], "lpips": [], "time_ms": []},
        "SRGAN": {"psnr": [], "ssim": [], "lpips": [], "time_ms": []},
        "ESRGAN": {"psnr": [], "ssim": [], "lpips": [], "time_ms": []}
    }

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[*] Evaluating {len(hr_files)} validation images...\n")

    for idx, hr_path in enumerate(hr_files):
        filename = os.path.basename(hr_path)
        lr_path = os.path.join(valid_lr_dir, filename)

        hr_pil = Image.open(hr_path).convert("RGB").resize((512, 512), Image.BICUBIC)
        if os.path.exists(lr_path):
            lr_pil = Image.open(lr_path).convert("RGB").resize((128, 128), Image.BICUBIC)
        else:
            lr_pil = hr_pil.resize((128, 128), Image.BICUBIC)

        hr_np = np.array(hr_pil)
        lr_np = np.array(lr_pil)

        lr_tensor = torch.tensor(lr_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        hr_tensor = torch.tensor(hr_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        # 1. Bicubic Baseline
        t0 = time.time()
        bicubic_np = run_bicubic_np(lr_np, hr_size=512)
        t_bicubic = (time.time() - t0) * 1000.0

        bicubic_tensor = torch.tensor(bicubic_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        results["Bicubic"]["psnr"].append(calculate_psnr(bicubic_tensor, hr_tensor))
        results["Bicubic"]["ssim"].append(calculate_ssim(bicubic_tensor, hr_tensor))
        results["Bicubic"]["lpips"].append(calculate_lpips(bicubic_tensor, hr_tensor, device=device))
        results["Bicubic"]["time_ms"].append(t_bicubic)

        # 2. SRGAN Baseline
        t0 = time.time()
        with torch.no_grad():
            srgan_tensor = srgan(lr_tensor)
        t_srgan = (time.time() - t0) * 1000.0

        results["SRGAN"]["psnr"].append(calculate_psnr(srgan_tensor, hr_tensor))
        results["SRGAN"]["ssim"].append(calculate_ssim(srgan_tensor, hr_tensor))
        results["SRGAN"]["lpips"].append(calculate_lpips(srgan_tensor, hr_tensor, device=device))
        results["SRGAN"]["time_ms"].append(t_srgan)

        # 3. ESRGAN System
        t0 = time.time()
        with torch.no_grad():
            esrgan_tensor = esrgan(lr_tensor)
        t_esrgan = (time.time() - t0) * 1000.0

        results["ESRGAN"]["psnr"].append(calculate_psnr(esrgan_tensor, hr_tensor))
        results["ESRGAN"]["ssim"].append(calculate_ssim(esrgan_tensor, hr_tensor))
        results["ESRGAN"]["lpips"].append(calculate_lpips(esrgan_tensor, hr_tensor, device=device))
        results["ESRGAN"]["time_ms"].append(t_esrgan)

        if idx == 0:
            esrgan_np = (esrgan_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            lr_upscaled = np.array(lr_pil.resize((512, 512), Image.NEAREST))
            comp_row = np.hstack([lr_upscaled, bicubic_np, esrgan_np, hr_np])
            out_img_path = os.path.join(args.output_dir, "comparison_matrix.png")
            Image.fromarray(comp_row).save(out_img_path)
            print(f"[+] Saved sample visual comparison to '{out_img_path}'")

    print(f"\n{'='*68}")
    print(f"{'Model / Method':<18} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'LPIPS':<8} | {'Latency (ms)':<12}")
    print(f"{'-'*68}")

    for model_name in ["Bicubic", "SRGAN", "ESRGAN"]:
        m = results[model_name]
        avg_psnr = np.mean(m["psnr"])
        avg_ssim = np.mean(m["ssim"])
        avg_lpips = np.mean(m["lpips"])
        avg_time = np.mean(m["time_ms"])
        print(f"{model_name:<18} | {avg_psnr:<10.2f} | {avg_ssim:<8.4f} | {avg_lpips:<8.4f} | {avg_time:<12.2f}")

    print(f"{'='*68}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESRGAN Evaluation & Benchmark Suite")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to checkpoints")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output comparison images")
    parser.add_argument("--num_blocks", type=int, default=4, help="RRDB blocks in ESRGAN model")
    args = parser.parse_args()

    benchmark_models(args)
