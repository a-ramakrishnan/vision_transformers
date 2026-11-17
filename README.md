# Diffusion Models From Scratch: Autoencoders, VAEs & DDPMs

[![Open Autoencoder & VAE In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/vision_transformers/blob/main/autoencoder_vae/autoencoder_vae.ipynb)
[![Open DDPM In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/vision_transformers/blob/main/ddpm/ddpm.ipynb)

## About

**`vision_transformers`** is a hands-on, from-scratch PyTorch learning repository exploring the mathematical foundations, architectural design, and empirical trade-offs of modern deep generative models—tracing the progression from **Convolutional Autoencoders** and **Variational Autoencoders (VAEs)** to **Denoising Diffusion Probabilistic Models (DDPMs)**.

### Why This Project Exists
- **First-Principles Implementation:** Every core component—encoder/decoder bottlenecks, the VAE reparameterization trick and ELBO objective, forward/reverse Markov diffusion kernels, sinusoidal timestep embeddings, and time-conditioned U-Nets with self-attention—is implemented cleanly in pure PyTorch without hiding behind high-level generative frameworks.
- **Multi-Dataset Scaling:** Built to scale seamlessly across **CIFAR-10** ($32 \times 32 \times 3$ natural RGB images), **EuroSAT** ($64 \times 64 \times 3$ Sentinel-2 satellite imagery), **STL-10** ($64 \times 64 \times 3$), **CelebA** ($64 \times 64 \times 3$), and **MNIST** ($32 \times 32 \times 1$).
- **Quantitative Benchmarking:** Includes an integrated `benchmark.py` suite inside each package to measure **Reconstruction Fidelity** (`MSE`, `PSNR`, `SSIM`), **Generative Quality** (`FID`, `Inception Score`), and **Compute Efficiency** (parameter count, training time/epoch, and sampling latency).
- **Colab-First Workflow:** Pre-configured for **NVIDIA A100 GPU** (with automatic **L4 GPU** fallback) in both modular packages and synchronized standalone notebooks.

The implementations reproduce and extend the foundations taught in Vizuara's deep-dive lectures:
1. [Autoencoder + VAE — Intuition + Coding from Scratch](https://www.youtube.com/watch?v=WMIiF-WQY9E&list=PLPTV0NXA_ZSgMaz0Mu-SjCPZNUjz6-6tN&index=9)
2. [Diffusion Model (DDPM) — Intuition + Coding from Scratch](https://www.youtube.com/watch?v=3Adu4riBh3Y&list=PLPTV0NXA_ZSgMaz0Mu-SjCPZNUjz6-6tN&index=2)

---

## Repository Structure

Both implementations are provided in two synchronized formats:
- **Modular packages** inside [`autoencoder_vae/`](./autoencoder_vae) and [`ddpm/`](./ddpm), each with its own Colab runner notebook, benchmark module, and saved training/benchmark results.
- **Standalone root notebooks** ([`autoencoder_vae.ipynb`](./autoencoder_vae.ipynb) and [`ddpm.ipynb`](./ddpm.ipynb)) kept in sync with all module changes.

```text
vision_transformers/
├── README.md
├── autoencoder_vae.ipynb              # Standalone root notebook (ConvAutoencoder + ConvVAE + Benchmarks)
├── ddpm.ipynb                         # Standalone root notebook (DDPM + Benchmarks)
├── autoencoder_vae/
│   ├── __init__.py
│   ├── dataset.py                     # Multi-dataset loader (CIFAR-10, EuroSAT, STL-10, CelebA, MNIST)
│   ├── models.py                      # ConvAutoencoder, ConvEncoder, ConvDecoder, ConvVAE & ELBO loss
│   ├── train.py                       # Training loops with epoch timing & metrics export
│   ├── visualize.py                   # RGB/grayscale reconstruction & prior-sampling grids
│   ├── benchmark.py                   # MSE, PSNR, SSIM, FID, Inception Score, Params & Latency
│   ├── autoencoder_vae.ipynb          # Modular Colab runner notebook (A100 / L4 GPU)
│   ├── training_results.json          # Saved epoch-by-epoch training metrics
│   └── benchmark_results.json         # Saved reconstruction, generative & compute benchmarks
└── ddpm/
    ├── __init__.py
    ├── dataset.py                     # Multi-dataset [-1, 1] loader (CIFAR-10, EuroSAT, STL-10, CelebA, MNIST)
    ├── scheduler.py                   # LinearNoiseScheduler (forward & reverse transition kernels)
    ├── unet.py                        # Sinusoidal time embeddings & Time-Conditioned U-Net
    ├── train.py                       # DDPM noise-prediction training loop with epoch timing
    ├── sample.py                      # Forward diffusion & reverse Markov sampling visualizations
    ├── benchmark.py                   # Partial-denoising MSE/PSNR/SSIM, FID, IS, Params & Latency
    ├── ddpm.ipynb                     # Modular Colab runner notebook (A100 / L4 GPU)
    ├── training_results.json          # Saved epoch-by-epoch noise MSE loss history
    └── benchmark_results.json         # Saved reconstruction, generative & compute benchmarks
```

---

## Supported Datasets

| Dataset | Resolution & Channels | Domain | `dataset_name` |
| :--- | :---: | :--- | :--- |
| **CIFAR-10** | $32 \times 32 \times 3$ RGB | Natural objects (50k train / 10k test) | `"cifar10"` |
| **EuroSAT** | $64 \times 64 \times 3$ RGB | Sentinel-2 satellite imagery (27k, 80/20 split) | `"eurosat"` |
| **STL-10** | $64 \times 64 \times 3$ RGB | Natural images (`train+unlabeled` / `test`) | `"stl10"` |
| **CelebA** | $64 \times 64 \times 3$ RGB | Celebrity faces (center-cropped & resized) | `"celeba"` |
| **MNIST** | $32 \times 32 \times 1$ Grayscale | Handwritten digits (60k train / 10k test) | `"mnist"` |

---

## Training in Google Colab

All notebooks are pre-configured in their Colab metadata to request an **NVIDIA A100 GPU** by default (`"gpuType": "A100"`), with **L4 GPU** as the automatic fallback if A100 capacity is unavailable (`Runtime -> Change runtime type -> A100 GPU` or `L4 GPU`).

| Implementation | Modular Runner Notebook | Standalone Root Notebook |
| :--- | :--- | :--- |
| **ConvAutoencoder + ConvVAE** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/vision_transformers/blob/main/autoencoder_vae/autoencoder_vae.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/vision_transformers/blob/main/autoencoder_vae.ipynb) |
| **DDPM** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/vision_transformers/blob/main/ddpm/ddpm.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/vision_transformers/blob/main/ddpm.ipynb) |

---

## Benchmark Suite

Each package's `benchmark.py` evaluates:
1. **Reconstruction Fidelity (Test Set):** `MSE`, `PSNR (dB)`, and `SSIM` (for DDPM, measures partial noising to $t = 250$ followed by reverse denoising to $t = 0$).
2. **Generative Quality (Prior Samples):** **FID** (Fréchet Inception Distance, lower is better) and **Inception Score** (higher is better) computed via pretrained `InceptionV3` features.
3. **Compute Efficiency:** Parameter count, model size (MB), average epoch time (sec), and sampling latency (`ms/image` and `images/sec`).

---

## Colab Benchmark Results (NVIDIA L4 GPU)

Full epoch-by-epoch training logs and benchmark JSONs are saved in [`autoencoder_vae/training_results.json`](./autoencoder_vae/training_results.json), [`autoencoder_vae/benchmark_results.json`](./autoencoder_vae/benchmark_results.json), [`ddpm/training_results.json`](./ddpm/training_results.json), and [`ddpm/benchmark_results.json`](./ddpm/benchmark_results.json).

### 1. CIFAR-10 ($32 \times 32 \times 3$ RGB)

| Model | Recon MSE ↓ | PSNR (dB) ↑ | SSIM ↑ | FID ↓ | Inception Score ↑ | Params | Avg Epoch | Sampling Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ConvAutoencoder`** | **`0.004792`** | **`23.20`** | **`0.8396`** | `361.21` | `1.143 ± 0.010` | `2.37M` | `3.81s` | `0.031 ms/img` |
| **`ConvVAE`** | `0.011958` | `19.22` | `0.6505` | `206.26` | `1.962 ± 0.067` | `2.90M` | `3.79s` | **`0.019 ms/img`** |
| **`DDPM`** ($T=1000$) | `0.008657` *(t=250)* | `20.63` | `0.7770` | **`115.65`** | **`3.854 ± 0.251`** | `6.29M` | `30.89s` | `5560.75 ms/img` |

### 2. EuroSAT ($64 \times 64 \times 3$ RGB Satellite Imagery)

| Model | Recon MSE ↓ | PSNR (dB) ↑ | SSIM ↑ | FID ↓ | Inception Score ↑ | Params | Avg Epoch | Sampling Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ConvAutoencoder`** | **`0.002436`** | **`26.15`** | **`0.7574`** | `301.62` | `1.069 ± 0.007` | `5.53M` | `5.00s` | `0.042 ms/img` |
| **`ConvVAE`** | `0.004080` | `23.90` | `0.6846` | `231.50` | `1.481 ± 0.057` | `7.63M` | `5.00s` | **`0.041 ms/img`** |
| **`DDPM`** ($T=1000$) | `0.006376` *(t=250)* | `21.96` | `0.7155` | **`114.66`** | **`3.314 ± 0.289`** | `6.29M` | `74.59s` | `16895.04 ms/img` |

### Key Empirical Takeaways
- **Reconstruction vs. Generation Trade-off:** `ConvAutoencoder` achieves the lowest pixel reconstruction error (`23.20 dB` / `26.15 dB` PSNR), but its unregularized latent space yields poor random samples (`FID > 300`).
- **Probabilistic Latent Regularization:** `ConvVAE` improves generative quality substantially (`FID 206.26` on CIFAR-10, `231.50` on EuroSAT) with sub-millisecond sampling latency.
- **Iterative Denoising Advantage:** `DDPM` achieves the best generative quality across both datasets (**FID `115.65` / `114.66`**, **IS `3.854` / `3.314`**) at the cost of `1000`-step sequential reverse sampling (`5.56s/img` at $32\times32$, `16.90s/img` at $64\times64$).

