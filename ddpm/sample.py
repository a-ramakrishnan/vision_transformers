import torch
import matplotlib.pyplot as plt


def plot_forward_diffusion(
    scheduler,
    train_loader,
    timesteps_to_show=None,
    device="cpu",
    save_path="forward_diffusion.png",
):
    if timesteps_to_show is None:
        timesteps_to_show = [0, 100, 300, 600, 999]

    sample_imgs, _ = next(iter(train_loader))
    x0 = sample_imgs[0:1].to(device)

    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(10, 2.5))
    for idx, step in enumerate(timesteps_to_show):
        t_tensor = torch.tensor([step], device=device)
        noise = torch.randn_like(x0)
        xt = scheduler.add_noise(x0, noise, t_tensor)
        img_display = (xt.cpu().squeeze() + 1.0) / 2.0
        axes[idx].imshow(img_display.clamp(0.0, 1.0), cmap="gray")
        axes[idx].set_title(f"t = {step}")
        axes[idx].axis("off")

    plt.suptitle("Forward Diffusion Process q(x_t | x_0)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def sample_ddpm(model, scheduler, num_samples=16, im_size=28, im_channels=1, device="cpu"):
    model.eval()
    xt = torch.randn((num_samples, im_channels, im_size, im_size), device=device)

    for i in reversed(range(scheduler.num_timesteps)):
        t_batch = torch.full((num_samples,), i, device=device, dtype=torch.long)
        noise_pred = model(xt, t_batch)
        xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, i)

    imgs = torch.clamp(xt, -1.0, 1.0).cpu()
    imgs = (imgs + 1.0) / 2.0
    return imgs


def plot_ddpm_samples(generated_imgs, save_path="ddpm_samples.png"):
    num_samples = generated_imgs.shape[0]
    cols = 8
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 1.5 * rows))
    for i in range(num_samples):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.imshow(generated_imgs[i].squeeze(), cmap="gray")
        ax.axis("off")

    plt.suptitle("DDPM: Generated MNIST Digits from Pure Gaussian Noise")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
