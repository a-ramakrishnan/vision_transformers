import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DATASET_CONFIGS = {
    "cifar10": {"in_channels": 3, "image_size": 32},
    "eurosat": {"in_channels": 3, "image_size": 64},
    "stl10": {"in_channels": 3, "image_size": 64},
    "celeba": {"in_channels": 3, "image_size": 64},
    "mnist": {"in_channels": 1, "image_size": 32},
}


def get_dataloaders(dataset_name="cifar10", batch_size=128, data_dir="./data", num_workers=2):
    """Load train and test DataLoaders for CIFAR-10, EuroSAT, STL-10, CelebA, or MNIST."""
    name = dataset_name.lower()
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose from {list(DATASET_CONFIGS.keys())}.")

    cfg = DATASET_CONFIGS[name]
    image_size = cfg["image_size"]

    if name == "celeba":
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    if name == "cifar10":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    elif name == "eurosat":
        full_dataset = datasets.EuroSAT(root=data_dir, transform=transform, download=True)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    elif name == "stl10":
        train_dataset = datasets.STL10(root=data_dir, split="train+unlabeled", transform=transform, download=True)
        test_dataset = datasets.STL10(root=data_dir, split="test", transform=transform, download=True)
    elif name == "celeba":
        train_dataset = datasets.CelebA(root=data_dir, split="train", transform=transform, download=True)
        test_dataset = datasets.CelebA(root=data_dir, split="test", transform=transform, download=True)
    else:
        train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, cfg
