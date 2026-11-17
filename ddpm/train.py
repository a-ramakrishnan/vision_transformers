import json
import time
import torch
import torch.nn as nn
import torch.optim as optim


def train_ddpm(
    model,
    scheduler,
    train_loader,
    num_epochs=20,
    lr=2e-4,
    device="cpu",
    checkpoint_path="ddpm.pth",
    results_path=None,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = []

    for epoch in range(num_epochs):
        t0 = time.time()
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
        epoch_time = time.time() - t0
        history.append({
            "epoch": epoch + 1,
            "noise_mse_loss": round(mean_loss, 6),
            "epoch_time_sec": round(epoch_time, 2),
        })
        print(f"[DDPM] Epoch [{epoch + 1}/{num_epochs}] - Noise MSE Loss: {mean_loss:.6f} ({epoch_time:.2f}s)")

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)

    if results_path:
        with open(results_path, "w") as f:
            json.dump({"ddpm": history}, f, indent=2)

    return model, history
