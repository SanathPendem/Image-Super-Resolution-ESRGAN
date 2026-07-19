import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import time
import requests
import base64
import numpy as np
from PIL import Image, ImageEnhance
import torch
import streamlit as st

try:
    from models.generator import RRDBNet
    from utils.metrics import calculate_psnr, calculate_ssim
    LOCAL_ENGINE_AVAILABLE = True
except Exception:
    LOCAL_ENGINE_AVAILABLE = False

st.set_page_config(
    page_title="ESRGAN Image Super-Resolution",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8F00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #A0AAB5;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ Research-Grade Image Super-Resolution (ESRGAN)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Industry-Standard 4× AI Upscaling Engine (128×128 → 512×512) with Real-Time Quality Benchmarking</div>', unsafe_allow_html=True)

st.sidebar.header("⚙️ System Configuration")
api_url = st.sidebar.text_input("FastAPI Endpoint URL", "http://localhost:8000")
device_option = st.sidebar.radio("Inference Device", ["Auto (CUDA/CPU)", "CPU Only"])
use_api = st.sidebar.checkbox("Use FastAPI Backend Server", value=True)
sharpness_boost = st.sidebar.slider("Edge Sharpening Intensity", min_value=1.0, max_value=3.5, value=2.2, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Architecture Spec")
st.sidebar.markdown("""
- **Model**: RRDBNet (ESRGAN)
- **Blocks**: 23 Residual-in-Residual Dense Blocks
- **Scaling Factor**: 4× Nearest + Conv
- **Loss Objective**: $L_1$ Pixel + VGG19 Perceptual + RaGAN
""")

@st.cache_resource
def load_local_model():
    if not LOCAL_ENGINE_AVAILABLE:
        return None, "cpu"
    device = torch.device("cpu" if device_option == "CPU Only" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = os.path.join("checkpoints", "esrgan_best.pth")
    num_features = 64
    num_blocks = 2
    state_dict = None

    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt.get("generator_state_dict", ckpt)
            if "conv_first.weight" in state_dict:
                num_features = state_dict["conv_first.weight"].shape[0]
            block_indices = [int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("rrdbs.")]
            if block_indices:
                num_blocks = max(block_indices) + 1
        except Exception:
            pass

    model = RRDBNet(num_features=num_features, num_blocks=num_blocks).to(device)
    if state_dict is not None:
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            pass
    model.eval()
    return model, device

def enhance_clarity(pil_img, boost_factor=2.2):
    sharp = ImageEnhance.Sharpness(pil_img).enhance(boost_factor)
    contrast = ImageEnhance.Contrast(sharp).enhance(1.15)
    return contrast

uploaded_files = st.file_uploader("Upload Image(s) (JPG / PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### 📷 Image Sample #{idx+1}: `{uploaded_file.name}`")
        pil_img = Image.open(uploaded_file).convert("RGB")

        # Downsample to 128x128 LR
        lr_pil = pil_img.resize((128, 128), Image.BICUBIC)
        lr_np = np.array(lr_pil)

        # Bicubic baseline (512x512)
        bicubic_pil = lr_pil.resize((512, 512), Image.BICUBIC)
        bicubic_np = np.array(bicubic_pil)

        sr_img_pil = None
        latency = 0.0
        psnr_val = 0.0
        ssim_val = 0.0

        if use_api:
            try:
                t0 = time.time()
                uploaded_file.seek(0)
                resp = requests.post(f"{api_url}/super-resolution", files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)})
                if resp.status_code == 200:
                    data = resp.json()
                    latency = data.get("latency_ms", 0.0)
                    psnr_val = data["metrics"]["psnr_vs_bicubic"]
                    ssim_val = data["metrics"]["ssim_vs_bicubic"]
                    
                    base64_str = data["enhanced_image_base64"].split(",")[1]
                    sr_bytes = base64.b64decode(base64_str)
                    raw_sr_pil = Image.open(io.BytesIO(sr_bytes))
                    sr_img_pil = enhance_clarity(raw_sr_pil, boost_factor=sharpness_boost)
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.warning(f"Could not connect to FastAPI Backend at '{api_url}'. Falling back to local PyTorch engine. Error: {e}")
                use_api = False

        if not use_api or sr_img_pil is None:
            model, device = load_local_model()
            if model is not None:
                lr_tensor = torch.tensor(lr_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                bicubic_tensor = torch.tensor(bicubic_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

                t0 = time.time()
                with torch.no_grad():
                    sr_tensor = model(lr_tensor)
                latency = (time.time() - t0) * 1000.0

                psnr_val = calculate_psnr(sr_tensor, bicubic_tensor)
                ssim_val = calculate_ssim(sr_tensor, bicubic_tensor)

                sr_np = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                raw_sr_pil = Image.fromarray(sr_np)
                sr_img_pil = enhance_clarity(raw_sr_pil, boost_factor=sharpness_boost)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(label="Inference Latency", value=f"{latency:.1f} ms")
        with col_m2:
            st.metric(label="PSNR Improvement", value=f"{psnr_val:.2f} dB")
        with col_m3:
            st.metric(label="SSIM Score", value=f"{ssim_val:.4f}")
        with col_m4:
            st.metric(label="Resolution Jump", value="128px ➔ 512px")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(lr_pil, caption="Low-Res Input (128×128)", use_container_width=True)
        with col2:
            st.image(bicubic_pil, caption="Bicubic Baseline (512×512)", use_container_width=True)
        with col3:
            if sr_img_pil is not None:
                st.image(sr_img_pil, caption="ESRGAN Enhanced (512×512)", use_container_width=True)

        st.markdown("---")

else:
    st.info("👆 Upload an image using the box above to evaluate 4× Super-Resolution upscaling.")
