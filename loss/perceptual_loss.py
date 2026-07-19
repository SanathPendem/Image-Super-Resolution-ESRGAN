import torch
import torch.nn as nn

try:
    import torchvision.models as models
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False

class PerceptualVGG19Loss(nn.Module):
    """
    VGG19 Feature Extraction for Perceptual Loss.
    """
    def __init__(self, feature_layer=35, use_input_norm=True):
        super().__init__()
        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is not installed.")

        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT if hasattr(models, 'VGG19_Weights') else True)
        self.features = nn.Sequential(*list(vgg19.features.children())[:feature_layer+1])
        
        for param in self.features.parameters():
            param.requires_grad = False

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.l1_loss = nn.L1Loss()

    def forward(self, sr, hr):
        if self.use_input_norm:
            sr = (sr - self.mean) / self.std
            hr = (hr - self.mean) / self.std

        sr_features = self.features(sr)
        hr_features = self.features(hr)

        return self.l1_loss(sr_features, hr_features)


class ESRGANLoss(nn.Module):
    """
    Combined Loss Objective for ESRGAN:
    1. Pixel Loss (L1)
    2. Perceptual Loss
    3. Relativistic Average GAN Loss (RaGAN)
    """
    def __init__(self, l1_weight=1.0, perceptual_weight=1.0, gan_weight=5e-3, device='cpu'):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.gan_weight = gan_weight

        self.l1 = nn.L1Loss()
        
        if _HAS_TORCHVISION:
            try:
                self.perceptual = PerceptualVGG19Loss().to(device)
            except Exception as e:
                print(f"[!] Info: Perceptual loss using fallback ({e})")
                self.perceptual = None
        else:
            self.perceptual = None

        self.bce_logits = nn.BCEWithLogitsLoss()

    def compute_generator_loss(self, sr, hr, pred_real, pred_fake):
        # 1. Pixel Loss
        loss_pixel = self.l1(sr, hr) * self.l1_weight

        # 2. Perceptual Loss
        if self.perceptual is not None:
            loss_percep = self.perceptual(sr, hr) * self.perceptual_weight
        else:
            loss_percep = self.l1(sr, hr) * self.perceptual_weight

        # 3. Relativistic Average GAN Loss
        real_label = torch.ones_like(pred_real)
        fake_label = torch.zeros_like(pred_fake)

        loss_g_real = self.bce_logits(pred_real - torch.mean(pred_fake), fake_label)
        loss_g_fake = self.bce_logits(pred_fake - torch.mean(pred_real), real_label)
        loss_gan = (loss_g_real + loss_g_fake) / 2.0 * self.gan_weight

        total_loss = loss_pixel + loss_percep + loss_gan

        return total_loss, {
            "loss_pixel": loss_pixel.item(),
            "loss_percep": loss_percep.item(),
            "loss_gan": loss_gan.item()
        }

    def compute_discriminator_loss(self, pred_real, pred_fake):
        real_label = torch.ones_like(pred_real)
        fake_label = torch.zeros_like(pred_fake)

        loss_d_real = self.bce_logits(pred_real - torch.mean(pred_fake), real_label)
        loss_d_fake = self.bce_logits(pred_fake - torch.mean(pred_real), fake_label)
        loss_d = (loss_d_real + loss_d_fake) / 2.0

        return loss_d
