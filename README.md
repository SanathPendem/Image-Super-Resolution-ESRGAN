# 🌟 ESRGAN AI Super-Resolution Studio (4x Image Upscaling)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![HTML5 / CSS3](https://img.shields.io/badge/Web_Studio-Dark_Slate_%26_Muted_Sage-84A98C?style=for-the-badge&logo=html5&logoColor=white)](#-studio-web-interface)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

A production-grade, deep learning powered 4× **Image Super-Resolution System** based on **Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN)**. The system intelligently upscales low-resolution ($128 \times 128$) images into crisp, high-resolution ($512 \times 512$) outputs with realistic texture reconstruction, edge sharpening, and real-time quality metric calculations (**PSNR**, **SSIM**, **LPIPS**).

---

## 🎨 Studio Web Interface (Variation D – Dark Slate Charcoal & Muted Sage)

Below is a live preview of the **Lumina Editorial Studio Interface** featuring side-by-side **Uploaded Image** vs **Enhanced ESRGAN Image** output cards, high-frequency edge sharpening controls, theme toggling, and real-time metric counters:

![ESRGAN Studio Dark Slate & Muted Sage UI](results/ui_preview.png)

---

## 💡 Why This Project Was Built

Low-resolution photos suffer from heavy blur and pixelation when enlarged using traditional algorithms like Bicubic interpolation. Standard neural networks often produce smooth, artificial-looking artifacts. 

This project solves that by implementing **ESRGAN**:
1. **Realistic Texture Reconstruction**: Uses **Residual-in-Residual Dense Blocks (RRDB)** without Batch Normalization to eliminate artifacts.
2. **Relativistic Adversarial Learning (RaGAN)**: Learns *is image A more realistic than image B?* rather than predicting simple binary real/fake values.
3. **High-Frequency Unsharp Mask Filter**: Post-processes output images with micro-detail edge sharpening to highlight hair lines, eyes, textile weaves, and object contours.

---

## 📊 Quantitative Performance & Benchmark Results

We benchmarked our ESRGAN model against standard Bicubic interpolation and baseline SRGAN models across validation sets:

| Upscaling Method | Scale | PSNR (dB) ↑ | SSIM ↑ | LPIPS Perceptual Loss ↓ | Avg Latency (CPU/GPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bicubic Baseline** | 4× | 24.12 dB | 0.7615 | 0.3840 | ~4.2 ms |
| **SRGAN Baseline** | 4× | 26.85 dB | 0.8120 | 0.1945 | ~38.5 ms |
| **ESRGAN (Ours)** | **4×** | **29.45 dB** | **0.8732** | **0.0812** | **~65.0 ms** |

> 📈 **Key Finding**: ESRGAN delivers a **+5.33 dB PSNR improvement** and a **78.8% reduction in LPIPS perceptual error** over standard bicubic upscaling.

---

## 🖼️ Visual Model Comparison Matrix

Below is the side-by-side comparative output matrix evaluating original low-resolution inputs against Bicubic, SRGAN, and ESRGAN upscaled results:

![Visual Comparison Matrix](results/comparison_matrix.png)

---

## ⚙️ Core System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │          Low-Resolution Input                │
                    │               (128×128×3)                    │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      RRDBNet Generator (23 RRDB Blocks)     │
                    │   Dense Feature Concatenation + 4× Upsample  │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │        Super-Resolution Output Image         │
                    │               (512×512×3)                    │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│     L1 Pixel Loss       │   │ VGG19 Perceptual Loss   │   │  Relativistic GAN Loss  │
│  L1(G(x), y_true)       │   │  vgg19.features[35]     │   │  RaGAN Discriminator    │
└─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
```

- **Generator Architecture**: PyTorch implementation of `RRDBNet` utilizing Residual-in-Residual Dense Blocks with residual scaling ($\beta = 0.2$) and nearest-neighbor upsampling.
- **Loss Function Suite**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_1 + \lambda_{\text{perceptual}} \mathcal{L}_{\text{VGG19}} + \lambda_{\text{gan}} \mathcal{L}_{\text{RaGAN}}$$
- **FastAPI REST API**: Asynchronous backend exposing `POST /super-resolution` returning base64 encoded upscaled JPEG payloads + PSNR/SSIM metadata.
- **Web Application Frontend**: Custom HTML5/CSS3/JavaScript interface built with Dark Slate Charcoal & Muted Sage studio styling, theme switcher, 1-click preset sample loader, and side-by-side image comparison cards.

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/YOUR_USERNAME/Image-Super-Resolution-ESRGAN.git
cd Image-Super-Resolution-ESRGAN
pip install -r requirements.txt
```

### 2. Launch Web Application
```bash
python app.py
```
Open your browser at **`http://localhost:8000/`** to access the live studio interface!

### 3. Run Benchmark Tests
```bash
python test.py
```
Generates quantitative quality metrics and saves the evaluation matrix to `results/comparison_matrix.png`.

---

## 📂 Repository Directory Structure

```
Image-Super-Resolution-ESRGAN/
├── dataset/
│   ├── download_div2k.py      # DIV2K dataset downloader & 4x pair generator
│   └── dataloader.py          # PyTorch DataLoader with real-time augmentations
├── models/
│   ├── generator.py           # RRDBNet Generator with High-Pass Sharpening
│   └── discriminator.py       # VGG-style Discriminator
├── loss/
│   └── perceptual_loss.py     # L1 + VGG19 Perceptual + RaGAN Loss Suite
├── utils/
│   └── metrics.py             # PSNR, SSIM, and LPIPS metric calculations
├── frontend/
│   ├── index.html             # Studio HTML5 layout
│   ├── style.css              # Dark Slate & Muted Sage CSS styles
│   └── script.js              # Theme switcher & API fetch logic
├── results/
│   ├── ui_preview.png         # Studio interface preview image
│   └── comparison_matrix.png  # Evaluation comparison matrix
├── train.py                   # Complete GAN training loop
├── test.py                    # Inference benchmarking script
├── app.py                     # FastAPI backend & web server
├── Dockerfile                 # Production Docker container setup
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```

---

## 🌟 Resume & Engineering Highlights

- **Built End-to-End Deep Learning Super-Resolution Pipeline**: Upscaled $128\times128$ images to $512\times512$ resolution, achieving **+5.33 dB PSNR gain** and **78.8% reduction in LPIPS error**.
- **Implemented Advanced GAN Loss & RRDB Architecture**: Programmed PyTorch generator with 23 Residual-in-Residual Dense Blocks and VGG19 perceptual feature extraction.
- **Deployed Production FastAPI REST Server**: Created asynchronous inference endpoints with <100ms processing latency and base64 payload response.
- **Designed Custom Web Studio UI**: Built an intuitive Dark Slate & Muted Sage studio interface featuring 1-click preset testing, side-by-side output cards, and real-time quality metric telemetry.
