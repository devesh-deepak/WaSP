import torch
import torch.nn.functional as F
import math


def rgb_to_ycbcr(img):
    """Converts RGB [0,1] to Y channel [0,1] using ITU-R BT.601."""
    y = 0.257 * img[:, 0, :, :] + 0.504 * img[:, 1, :, :] + 0.098 * img[:, 2, :, :] + 16 / 255.0
    return y.unsqueeze(1)


def calculate_psnr(img1, img2, crop_border=4):
    """
    Calculates PSNR on the Y channel.
    Args:
        img1, img2: Tensors [B, 3, H, W] in range [0, 1]
        crop_border: pixels to crop from edges (avoids padding artifacts)
    Returns:
        PSNR value in dB (scalar tensor).
    """
    img1_y = rgb_to_ycbcr(img1)
    img2_y = rgb_to_ycbcr(img2)

    if crop_border > 0:
        img1_y = img1_y[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2_y = img2_y[:, :, crop_border:-crop_border, crop_border:-crop_border]

    mse = F.mse_loss(img1_y, img2_y)
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(1.0 / mse)


def calculate_ssim(img1, img2, window_size=11, sigma=1.5):
    """
    Calculates SSIM on the Y channel.
    Args:
        img1, img2: Tensors [B, 3, H, W] in range [0, 1]
    Returns:
        Mean SSIM value (scalar tensor).
    """
    img1 = rgb_to_ycbcr(img1)
    img2 = rgb_to_ycbcr(img2)

    channel = 1
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
         for x in range(window_size)],
        device=img1.device
    )
    gauss = gauss / gauss.sum()
    window = (gauss.unsqueeze(1) * gauss.unsqueeze(0)).float()
    window = window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size)

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()