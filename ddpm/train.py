import torch
import torch.nn as nn
import torch.optim as optim


def train_ddpm(
    model,
    scheduler,
    train_loader,
    num_epochs=20,
    lr=1e-4,
    device="cpu",
    checkpoint_path="ddpm_mnist.pth",
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for images, _ in train_loader:
            images = images.to(device)
            noise = torch.randn_like(images)
            t = torch.randint(0, scheduler.num_timesteps, (images.shape[0],), device=device)

            noisy_images = scheduler.add_noise(images, noise, t)
            noise_pred = model(noisy_images, t)

            loss = criterion(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        print(f"[DDPM] Epoch [{epoch + 1}/{num_epochs}] - Noise MSE Loss: {mean_loss:.6f}")

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)

    return model
