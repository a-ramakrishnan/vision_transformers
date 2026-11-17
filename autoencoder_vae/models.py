import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder supporting RGB/grayscale images (32x32 or 64x64)."""

    def __init__(self, in_channels=3, image_size=32, latent_dim=128, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.feat_size = image_size // 8
        self.flat_dim = base_channels * 4 * self.feat_size * self.feat_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.base_channels = base_channels

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_enc(h.flatten(1))

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.base_channels * 4, self.feat_size, self.feat_size)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class ConvEncoder(nn.Module):
    """Convolutional VAE Encoder mapping images to (mu, logvar)."""

    def __init__(self, in_channels=3, image_size=32, latent_dim=128, base_channels=64):
        super().__init__()
        self.feat_size = image_size // 8
        self.flat_dim = base_channels * 4 * self.feat_size * self.feat_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConvDecoder(nn.Module):
    """Convolutional VAE Decoder reconstructing images from latent vector z."""

    def __init__(self, in_channels=3, image_size=32, latent_dim=128, base_channels=64):
        super().__init__()
        self.base_channels = base_channels
        self.feat_size = image_size // 8
        self.flat_dim = base_channels * 4 * self.feat_size * self.feat_size

        self.fc = nn.Linear(latent_dim, self.flat_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, self.base_channels * 4, self.feat_size, self.feat_size)
        return self.deconv(h)


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder with reparameterization trick."""

    def __init__(self, in_channels=3, image_size=32, latent_dim=128, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(in_channels, image_size, latent_dim, base_channels)
        self.decoder = ConvDecoder(in_channels, image_size, latent_dim, base_channels)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar, kld_weight=1.0):
    """Per-sample ELBO loss = pixel-wise BCE + weighted KL divergence."""
    batch_size = x.size(0)
    recon_clamped = torch.clamp(recon_x, 1e-6, 1.0 - 1e-6)
    x_clamped = torch.clamp(x, 0.0, 1.0)
    bce = F.binary_cross_entropy(recon_clamped, x_clamped, reduction="sum") / batch_size
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return bce + kld_weight * kld, bce, kld
