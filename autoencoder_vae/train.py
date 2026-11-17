import json
import time
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .models import vae_loss_function
except ImportError:
    from models import vae_loss_function


def train_autoencoder(
    model,
    train_loader,
    epochs=15,
    lr=1e-3,
    device="cpu",
    checkpoint_path="autoencoder.pth",
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            recon_images, _ = model(images)
            loss = criterion(recon_images, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        epoch_time = time.time() - t0
        history.append({
            "epoch": epoch + 1,
            "mse_loss": round(avg_loss, 6),
            "epoch_time_sec": round(epoch_time, 2),
        })
        print(f"[Autoencoder] Epoch [{epoch + 1}/{epochs}] - MSE Loss: {avg_loss:.6f} ({epoch_time:.2f}s)")

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)

    return model, history


def train_vae(
    model,
    train_loader,
    epochs=20,
    lr=1e-3,
    kld_weight=1.0,
    device="cpu",
    checkpoint_path="vae.pth",
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_kld = 0.0

        for images, _ in train_loader:
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss, bce, kld = vae_loss_function(recon_images, images, mu, logvar, kld_weight=kld_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_bce += bce.item() * bs
            total_kld += kld.item() * bs

        n_samples = len(train_loader.dataset)
        avg_total = total_loss / n_samples
        avg_bce = total_bce / n_samples
        avg_kld = total_kld / n_samples
        epoch_time = time.time() - t0
        history.append({
            "epoch": epoch + 1,
            "total_loss": round(avg_total, 4),
            "bce_loss": round(avg_bce, 4),
            "kld_loss": round(avg_kld, 4),
            "epoch_time_sec": round(epoch_time, 2),
        })
        print(
            f"[VAE] Epoch [{epoch + 1}/{epochs}] - "
            f"Total Loss: {avg_total:.4f} | "
            f"BCE: {avg_bce:.4f} | "
            f"KLD: {avg_kld:.4f} ({epoch_time:.2f}s)"
        )

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)

    return model, history


def save_training_results(results_dict, save_path="training_results.json"):
    with open(save_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    return results_dict
