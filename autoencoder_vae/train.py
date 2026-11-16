import torch.nn as nn
import torch.optim as optim

try:
    from .models import vae_loss_function
except ImportError:
    from models import vae_loss_function


def train_autoencoder(model, train_loader, epochs=10, lr=1e-3, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        print(f"[Autoencoder] Epoch [{epoch + 1}/{epochs}] - MSE Loss: {avg_loss:.6f}")

    return model


def train_vae(model, train_loader, epochs=15, lr=1e-3, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        print(
            f"[VAE] Epoch [{epoch + 1}/{epochs}] - "
            f"Total Loss: {total_loss / n_samples:.4f} | "
            f"BCE: {total_bce / n_samples:.4f} | "
            f"KLD: {total_kld / n_samples:.4f}"
        )

    return model
