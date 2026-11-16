# Diffusion Models From Scratch: Autoencoders, VAEs & DDPMs

[![Open Autoencoder & VAE In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/vision_transformers/blob/main/autoencoder_vae/autoencoder_vae.ipynb)
[![Open DDPM In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/vision_transformers/blob/main/ddpm/ddpm.ipynb)

## About

This repository is a hands-on, from-scratch PyTorch learning implementation of **Vanilla Autoencoders**, **Variational Autoencoders (VAEs)**, and **Denoising Diffusion Probabilistic Models (DDPMs)** on the MNIST dataset. It is designed to build a first-principles mathematical and architectural understanding of how modern generative diffusion models evolve from deterministic bottleneck autoencoders and probabilistic latent variable models.

The implementations reproduce and modularize the architectures taught in Vizuara's deep-dive lectures:
1. [Autoencoder + VAE — Intuition + Coding from Scratch](https://www.youtube.com/watch?v=WMIiF-WQY9E&list=PLPTV0NXA_ZSgMaz0Mu-SjCPZNUjz6-6tN&index=9)
2. [Diffusion Model (DDPM) — Intuition + Coding from Scratch](https://www.youtube.com/watch?v=3Adu4riBh3Y&list=PLPTV0NXA_ZSgMaz0Mu-SjCPZNUjz6-6tN&index=2)

---

## Repository Structure

Both implementations are provided in two synchronized formats:
- **Modular packages** inside [`autoencoder_vae/`](./autoencoder_vae) and [`ddpm/`](./ddpm), each with its own Colab runner notebook and saved training results.
- **Standalone root notebooks** ([`autoencoder_vae.ipynb`](./autoencoder_vae.ipynb) and [`ddpm.ipynb`](./ddpm.ipynb)) containing the complete self-contained code in a single file.

```text
vision_transformers/
├── README.md
├── autoencoder_vae.ipynb              # Standalone root notebook (Autoencoder + VAE)
├── ddpm.ipynb                         # Standalone root notebook (DDPM)
├── autoencoder_vae/
│   ├── __init__.py
│   ├── dataset.py                     # MNIST dataset & DataLoader setup
│   ├── models.py                      # Autoencoder, Encoder, Decoder, VAE & ELBO loss
│   ├── train.py                       # Training loops & metrics export
│   ├── visualize.py                   # Reconstruction & prior sampling plots
│   ├── autoencoder_vae.ipynb          # Modular Colab runner notebook (A100 / L4 GPU)
│   └── training_results.json          # Saved Colab training metrics (NVIDIA L4 GPU)
└── ddpm/
    ├── __init__.py
    ├── dataset.py                     # MNIST [-1, 1] scaled dataset & DataLoader
    ├── scheduler.py                   # LinearNoiseScheduler (forward & reverse kernels)
    ├── unet.py                        # Sinusoidal time embeddings & Time-Conditioned U-Net
    ├── train.py                       # DDPM noise-prediction training loop & metrics export
    ├── sample.py                      # Reverse Markov chain sampling & visualization
    ├── ddpm.ipynb                     # Modular Colab runner notebook (A100 / L4 GPU)
    └── training_results.json          # Saved Colab training metrics (NVIDIA A100-SXM4-80GB GPU)
```

---

## Models & Mathematical Foundations

### 1. Vanilla Autoencoder & Variational Autoencoder (`autoencoder_vae/`)

- **Vanilla Autoencoder:** Compresses $28 \times 28$ MNIST images ($784$-D) through a deterministic MLP bottleneck (`784 -> 256 -> 64 -> 2`) and reconstructs them via `MSELoss`. While effective for compression, deterministic latent spaces contain gaps that prevent coherent random sampling.
- **Variational Autoencoder (VAE):** Maps each input $x$ to a probabilistic Gaussian distribution parameterized by mean $\mu$ and log-variance $\log \sigma^2$:
  - **Reparameterization Trick:** Samples $z = \mu + \sigma \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$ so gradients flow through stochastic sampling.
  - **ELBO Objective:** Combines pixel-wise Binary Cross-Entropy reconstruction loss and closed-form Kullback-Leibler divergence:
    $$\mathcal{L}_{\text{VAE}} = \text{BCE}(x, \hat{x}) - \frac{1}{2} \sum \left(1 + \log \sigma^2 - \mu^2 - \sigma^2\right)$$

### 2. Denoising Diffusion Probabilistic Model (`ddpm/`)

- **Forward Process (`LinearNoiseScheduler.add_noise`):** Progressively corrupts clean images $x_0 \in [-1, 1]$ over $T = 1000$ timesteps using a linear variance schedule $\beta_t \in [10^{-4}, 0.02]$. Using $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, noisy samples at any timestep $t$ are drawn in closed form:
  $$q(x_t \mid x_0) = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
- **Time-Conditioned U-Net (`UNet`):** Predicts the injected noise $\epsilon_\theta(x_t, t)$ using `DownBlock`, `MidBlock`, and `UpBlock` stages equipped with `GroupNorm`, `SiLU` activations, residual connections, `MultiheadAttention`, and sinusoidal timestep embeddings.
- **Reverse Process (`LinearNoiseScheduler.sample_prev_timestep`):** Starts from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$ and iteratively denoises from $t = T-1$ down to $0$:
  $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

---

## Training in Google Colab

All notebooks are pre-configured in their Colab metadata to request an **NVIDIA A100 GPU** by default (`"gpuType": "A100"`), with **L4 GPU** as the automatic fallback if A100 capacity is unavailable (`Runtime -> Change runtime type -> A100 GPU` or `L4 GPU`).

| Implementation | Modular Runner Notebook | Standalone Root Notebook |
| :--- | :--- | :--- |
| **Autoencoder + VAE** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/vision_transformers/blob/main/autoencoder_vae/autoencoder_vae.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/vision_transformers/blob/main/autoencoder_vae.ipynb) |
| **DDPM** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/vision_transformers/blob/main/ddpm/ddpm.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/vision_transformers/blob/main/ddpm.ipynb) |

---

## Saved Colab Training Results

Full epoch-by-epoch training metrics are stored in [`autoencoder_vae/training_results.json`](./autoencoder_vae/training_results.json) and [`ddpm/training_results.json`](./ddpm/training_results.json).

### 1. Autoencoder & VAE (`autoencoder_vae/` — Trained on NVIDIA L4 GPU)

| Model | Epochs | Initial Loss (Epoch 1) | Final Loss |
| :--- | :---: | :---: | :---: |
| **Vanilla Autoencoder** (`latent_dim=2`) | 10 | `0.061032` (MSE) | `0.039341` (MSE) |
| **Variational Autoencoder** (`latent_dim=20`) | 15 | `165.1622` (`BCE: 149.8300`, `KLD: 15.3322`) | `104.7431` (`BCE: 79.4418`, `KLD: 25.3013`) |

### 2. DDPM (`ddpm/` — Trained on NVIDIA A100-SXM4-80GB High RAM GPU)

| Epoch | Noise Prediction MSE Loss |
| :---: | :---: |
| 1 | `0.139174` |
| 5 | `0.030304` |
| 10 | `0.026757` |
| 15 | `0.025046` |
| 20 | `0.024263` |
