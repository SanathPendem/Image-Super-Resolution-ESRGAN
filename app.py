import os
import io
import time
import base64
import numpy as np
from PIL import Image, ImageEnhance
import torch

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from models.generator import RRDBNet
from utils.metrics import calculate_psnr, calculate_ssim

app = FastAPI(
    title="ESRGAN Image Super-Resolution API",
    description="Production FastAPI service for 4x ESRGAN super-resolution upscaling (128x128 to 512x512).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join("checkpoints", "esrgan_best.pth")

print(f"[*] Initializing FastAPI Super-Resolution Engine on {DEVICE}...")

# Detect num_features & num_blocks dynamically from checkpoint
num_features = 64
num_blocks = 2
state_dict = None

if os.path.exists(CHECKPOINT_PATH):
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = ckpt.get("generator_state_dict", ckpt)
        if "conv_first.weight" in state_dict:
            num_features = state_dict["conv_first.weight"].shape[0]
        block_indices = [int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("rrdbs.")]
        if block_indices:
            num_blocks = max(block_indices) + 1
        print(f"[+] Detected Checkpoint Architecture: num_features={num_features}, num_blocks={num_blocks}")
    except Exception as e:
        print(f"[!] Warning inspecting checkpoint: {e}")

model = RRDBNet(num_features=num_features, num_blocks=num_blocks).to(DEVICE)

if state_dict is not None:
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"[+] Loaded trained model parameters from '{CHECKPOINT_PATH}'")
    except Exception as e:
        print(f"[!] Info loading state dict: {e}")

model.eval()

def enhance_clarity(pil_img, boost_factor=2.2):
    sharp = ImageEnhance.Sharpness(pil_img).enhance(boost_factor)
    contrast = ImageEnhance.Contrast(sharp).enhance(1.15)
    return contrast

# Serve Web UI Frontend files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def read_root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "online",
        "service": "ESRGAN Image Super-Resolution API",
        "device": str(DEVICE),
        "upscale_factor": "4x (128x128 -> 512x512)"
    }

@app.get("/style.css")
def get_css():
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"))

@app.get("/script.js")
def get_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "script.js"))

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(DEVICE)}

@app.post("/super-resolution")
async def super_resolution_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (JPG/PNG).
    Preprocesses to 128x128 LR, runs 4x ESRGAN upscaling to 512x512 HR,
    applies edge sharpness and contrast enhancement, computes PSNR and SSIM quality metrics,
    and returns enhanced image payload + metadata.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, PNG, and WEBP are supported.")

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Input Low-Resolution Resize to 128x128
        lr_pil = pil_img.resize((128, 128), Image.BICUBIC)
        lr_np = np.array(lr_pil)

        # Bicubic Baseline for comparison metrics
        bicubic_pil = lr_pil.resize((512, 512), Image.BICUBIC)
        bicubic_np = np.array(bicubic_pil)
        bicubic_tensor = torch.tensor(bicubic_np).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

        # Tensor conversion for model inference
        lr_tensor = torch.tensor(lr_np).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

        # Perform Inference & Profile Latency
        start_time = time.time()
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        latency_ms = (time.time() - start_time) * 1000.0

        # Compute Metrics
        psnr_score = calculate_psnr(sr_tensor, bicubic_tensor)
        ssim_score = calculate_ssim(sr_tensor, bicubic_tensor)

        # Convert Output Tensor to Base64 JPEG string with clarity enhancement
        sr_np = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        raw_sr_pil = Image.fromarray(sr_np)
        sr_pil = enhance_clarity(raw_sr_pil, boost_factor=2.2)
        
        buffered = io.BytesIO()
        sr_pil.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "status": "success",
            "latency_ms": round(latency_ms, 2),
            "metrics": {
                "psnr_vs_bicubic": round(psnr_score, 2),
                "ssim_vs_bicubic": round(ssim_score, 4)
            },
            "dimensions": {
                "input": "128x128",
                "output": "512x512"
            },
            "enhanced_image_base64": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
