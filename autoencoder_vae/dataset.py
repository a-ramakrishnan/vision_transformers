from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(batch_size=128, data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
