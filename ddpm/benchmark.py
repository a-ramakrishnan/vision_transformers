import json
import math
import time
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

try:
    from .sample import sample_ddpm
except ImportError:
    from sample import sample_ddpm


def _gaussian_window(window_size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim(img1, img2, window_size=11):
    """Compute mean SSIM between two [0, 1] image batches."""
    channels = img1.size(1)
    window = _gaussian_window(window_size, 1.5, channels, img1.device)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


@torch.no_grad()
def evaluate_ddpm_reconstruction(model, scheduler, test_loader, t_noise=250, device="cpu", max_samples=128):
    """Corrupt test images to timestep t_noise and denoise back to t=0 to measure MSE, PSNR, and SSIM."""
    model.eval()
    mse_list, psnr_list, ssim_list = [], [], []
    seen = 0

    for images, _ in test_loader:
        images = images.to(device)
        noise = torch.randn_like(images)
        t_tensor = torch.full((images.size(0),), t_noise, device=device, dtype=torch.long)
        xt = scheduler.add_noise(images, noise, t_tensor)

        for step in reversed(range(t_noise + 1)):
            t_batch = torch.full((images.size(0),), step, device=device, dtype=torch.long)
            noise_pred = model(xt, t_batch)
            xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, step)

        orig_01 = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
        recon_01 = torch.clamp((xt + 1.0) / 2.0, 0.0, 1.0)

        mse = F.mse_loss(recon_01, orig_01, reduction="mean").item()
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))
        ssim = compute_ssim(orig_01, recon_01)

        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        seen += images.size(0)
        if seen >= max_samples:
            break

    return {
        "t_noise_level": t_noise,
        "mse": round(float(np.mean(mse_list)), 6),
        "psnr_db": round(float(np.mean(psnr_list)), 3),
        "ssim": round(float(np.mean(ssim_list)), 4),
    }


def _load_inception_models(device):
    weights = Inception_V3_Weights.DEFAULT
    clf = inception_v3(weights=weights).to(device).eval()
    feat_extractor = inception_v3(weights=weights).to(device).eval()
    feat_extractor.fc = nn.Identity()
    return feat_extractor, clf


def _preprocess_for_inception(imgs):
    if imgs.size(1) == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
    return (imgs - mean) / std


@torch.no_grad()
def _extract_inception_stats(images_tensor, feat_extractor, clf, batch_size=64, device="cpu"):
    feats, probs = [], []
    for i in range(0, images_tensor.size(0), batch_size):
        batch = images_tensor[i : i + batch_size].to(device)
        inp = _preprocess_for_inception(batch)
        feats.append(feat_extractor(inp).cpu().numpy())
        logits = clf(inp)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(feats, axis=0), np.concatenate(probs, axis=0)


def compute_fid(real_feats, gen_feats):
    mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = np.mean(gen_feats, axis=0), np.cov(gen_feats, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return round(fid, 3)


def compute_inception_score(probs, splits=5):
    scores = []
    n = probs.shape[0]
    split_size = max(1, n // splits)
    for k in range(splits):
        part = probs[k * split_size : (k + 1) * split_size]
        if len(part) == 0:
            continue
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return round(float(np.mean(scores)), 3), round(float(np.std(scores)), 3)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "model_size_mb": round(size_mb, 3),
    }


@torch.no_grad()
def benchmark_ddpm(
    model,
    scheduler,
    test_loader,
    im_size=32,
    im_channels=3,
    device="cpu",
    history=None,
    num_gen_samples=128,
    batch_size=64,
    t_noise=250,
):
    """Run Reconstruction (partial denoising), Generative Quality (FID/IS), and Compute Efficiency benchmarks."""
    model.eval()
    recon_metrics = evaluate_ddpm_reconstruction(
        model, scheduler, test_loader, t_noise=t_noise, device=device, max_samples=batch_size
    )
    param_metrics = count_parameters(model)

    gen_batches = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.time()
    for i in range(0, num_gen_samples, batch_size):
        cur_bs = min(batch_size, num_gen_samples - i)
        imgs = sample_ddpm(model, scheduler, num_samples=cur_bs, im_size=im_size, im_channels=im_channels, device=device)
        gen_batches.append(imgs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t_start
    gen_tensor = torch.cat(gen_batches, dim=0)

    real_batches = []
    collected = 0
    for imgs, _ in test_loader:
        real_batches.append(torch.clamp((imgs + 1.0) / 2.0, 0.0, 1.0))
        collected += imgs.size(0)
        if collected >= num_gen_samples:
            break
    real_tensor = torch.cat(real_batches, dim=0)[:num_gen_samples]

    feat_extractor, clf = _load_inception_models(device)
    real_feats, _ = _extract_inception_stats(real_tensor, feat_extractor, clf, batch_size=64, device=device)
    gen_feats, gen_probs = _extract_inception_stats(gen_tensor, feat_extractor, clf, batch_size=64, device=device)

    fid = compute_fid(real_feats, gen_feats)
    is_mean, is_std = compute_inception_score(gen_probs)

    avg_epoch_time = (
        round(float(np.mean([h["epoch_time_sec"] for h in history])), 2) if history else None
    )

    return {
        "reconstruction": recon_metrics,
        "generative_quality": {
            "fid": fid,
            "inception_score_mean": is_mean,
            "inception_score_std": is_std,
            "num_eval_samples": num_gen_samples,
        },
        "compute_efficiency": {
            **param_metrics,
            "avg_epoch_time_sec": avg_epoch_time,
            "sampling_latency_ms_per_image": round((elapsed / num_gen_samples) * 1000.0, 4),
            "sampling_throughput_imgs_per_sec": round(num_gen_samples / max(elapsed, 1e-6), 2),
        },
    }


def save_benchmark_results(results, save_path="benchmark_results.json"):
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
