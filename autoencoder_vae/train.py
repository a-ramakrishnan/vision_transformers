import json
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
    epochs=10,
    lr=1e-3,
    device="cpu",
    checkpoint_path="autoencoder_mnist.pth",
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
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
        history.append({"epoch": epoch + 1, "mse_loss": round(avg_loss, 6)})
        print(f"[Autoencoder] Epoch [{epoch + 1}/{epochs}] - MSE Loss: {avg_loss:.6f}")

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)

    return model, history


def train_vae(
    model,
    train_loader,
    epochs=15,
    lr=1e-3,
    device="cpu",
    checkpoint_path="vae_mnist.pth",
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_kld = 0.0

        for images, _ in train_loader:
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss, bce, kld = vae_loss_function(recon_images, images, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

        n_samples = len(train_loader.dataset)
        avg_total = total_loss / n_samples
        avg_bce = total_bce / n_samples
        avg_kld = total_kld / n_samples
        history.append({
            "epoch": epoch + 1,
            "total_loss": round(avg_total, 4),
            "bce_loss": round(avg_bce, 4),
            "kld_loss": round(avg_kld, 4),
        })
        print(
            f"[VAE] Epoch [{epoch + 1}/{epochs}] - "
            f"Total Loss: {avg_total:.4f} | "
            f"BCE: {avg_bce:.4f} | "
            f"KLD: {avg_kld:.4f}"
        )

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)

    return model, history


def save_training_results(ae_history, vae_history, save_path="training_results.json"):
    results = {
        "autoencoder": ae_history,
        "vae": vae_history,
    }
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
