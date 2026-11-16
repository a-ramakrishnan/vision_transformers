from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_ddpm_dataloader(batch_size=128, data_dir="./data"):
    # Scale MNIST images from [0, 1] to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2.0) - 1.0),
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader
