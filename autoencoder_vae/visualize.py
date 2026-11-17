import torch
import matplotlib.pyplot as plt


def _to_display_image(tensor):
    img = tensor.cpu()
    if img.shape[0] == 1:
        return img.squeeze(0), "gray"
    return img.permute(1, 2, 0).clamp(0.0, 1.0), None


@torch.no_grad()
def plot_reconstructions(
    model,
    test_loader,
    device="cpu",
    num_images=8,
    is_vae=False,
    title=None,
    save_path=None,
):
    model.eval()
    test_images, _ = next(iter(test_loader))
    test_images = test_images[:num_images].to(device)

    if is_vae:
        recon_images, _, _ = model(test_images)
    else:
        recon_images, _ = model(test_images)

    fig, axes = plt.subplots(2, num_images, figsize=(12, 3))
    for i in range(num_images):
        orig_img, cmap = _to_display_image(test_images[i])
        rec_img, _ = _to_display_image(recon_images[i])
        axes[0, i].imshow(orig_img, cmap=cmap)
        axes[0, i].axis("off")
        axes[1, i].imshow(rec_img, cmap=cmap)
        axes[1, i].axis("off")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def plot_vae_samples(
    vae_model,
    latent_dim=128,
    num_samples=16,
    device="cpu",
    save_path="vae_samples.png",
    title="VAE: Generated Samples from Prior z ~ N(0, I)",
):
    vae_model.eval()
    z_random = torch.randn(num_samples, latent_dim, device=device)
    generated_images = vae_model.decoder(z_random)

    cols = 8
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 1.5 * rows))
    for i in range(num_samples):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        img, cmap = _to_display_image(generated_images[i])
        ax.imshow(img, cmap=cmap)
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
