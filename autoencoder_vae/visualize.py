import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def plot_reconstructions(model, test_loader, device="cpu", num_images=8, is_vae=False, title=None):
    model.eval()
    test_images, _ = next(iter(test_loader))
    test_images = test_images[:num_images].to(device)

    if is_vae:
        recon_flat, _, _ = model(test_images)
        recon_images = recon_flat.view(-1, 1, 28, 28)
    else:
        recon_images, _ = model(test_images)

    fig, axes = plt.subplots(2, num_images, figsize=(12, 3))
    for i in range(num_images):
        axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_images[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_vae_samples(vae_model, latent_dim=20, num_samples=16, device="cpu"):
    vae_model.eval()
    z_random = torch.randn(num_samples, latent_dim, device=device)
    generated_images = vae_model.decoder(z_random).view(-1, 1, 28, 28)

    cols = 8
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 1.5 * rows))
    for i in range(num_samples):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.imshow(generated_images[i].cpu().squeeze(), cmap="gray")
        ax.axis("off")

    plt.suptitle("VAE: Generated Digits Sampled from Prior z ~ N(0, I)")
    plt.tight_layout()
    plt.show()
