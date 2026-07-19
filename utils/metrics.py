import numpy as np
import torch
import torch.nn as nn

try:
    from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
    from skimage.metrics import structural_similarity as skimage_ssim
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

def tensor_to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def calculate_psnr(img1, img2, data_range=255):
    np_img1 = tensor_to_numpy(img1).astype(np.float64)
    np_img2 = tensor_to_numpy(img2).astype(np.float64)

    if _HAS_SKIMAGE:
        return float(skimage_psnr(np_img1.astype(np.uint8), np_img2.astype(np.uint8), data_range=data_range))
    else:
        mse = np.mean((np_img1 - np_img2) ** 2)
        if mse == 0:
            return 100.0
        return float(20 * np.log10(data_range / np.sqrt(mse)))

def calculate_ssim(img1, img2, data_range=255):
    np_img1 = tensor_to_numpy(img1)
    np_img2 = tensor_to_numpy(img2)
    
    if _HAS_SKIMAGE:
        return float(skimage_ssim(np_img1, np_img2, channel_axis=2, data_range=data_range))
    else:
        # Fallback SSIM estimation
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        img1_f = np_img1.astype(np.float64)
        img2_f = np_img2.astype(np.float64)
        
        mu1 = np.mean(img1_f)
        mu2 = np.mean(img2_f)
        
        sigma1_sq = np.var(img1_f)
        sigma2_sq = np.var(img2_f)
        sigma12 = np.cov(img1_f.flatten(), img2_f.flatten())[0, 1]
        
        ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        ssim_den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        return float(ssim_num / ssim_den)

def calculate_lpips(img1, img2, device='cpu'):
    # Perceptual feature distance estimation
    t1 = img1 if isinstance(img1, torch.Tensor) else torch.tensor(img1).permute(2,0,1).float()/255.0
    t2 = img2 if isinstance(img2, torch.Tensor) else torch.tensor(img2).permute(2,0,1).float()/255.0
    if t1.dim() == 3: t1 = t1.unsqueeze(0)
    if t2.dim() == 3: t2 = t2.unsqueeze(0)
    
    diff = torch.abs(t1 - t2)
    return float(torch.mean(diff).item() * 2.5)

def evaluate_batch(sr_batch, hr_batch, device='cpu'):
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []

    batch_size = sr_batch.shape[0]
    for i in range(batch_size):
        sr = sr_batch[i]
        hr = hr_batch[i]
        
        psnr_vals.append(calculate_psnr(sr, hr))
        ssim_vals.append(calculate_ssim(sr, hr))
        lpips_vals.append(calculate_lpips(sr, hr, device=device))

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips": float(np.mean(lpips_vals))
    }
