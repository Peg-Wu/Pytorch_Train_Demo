import os
import torch
import torchvision as tv


def get_mnist_datasets(save_dir):
    r"""(C, H, W) --> (1, 28, 28)"""
    os.makedirs(save_dir, exist_ok=True)
    transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307), (0.3081))])
    train_ds = tv.datasets.MNIST(root=save_dir, train=True, transform=transform, download=True)
    valid_ds = tv.datasets.MNIST(root=save_dir, train=False, transform=transform, download=True)
    return train_ds, valid_ds


def collate_flatten_mnist(batch_list):
    r"""(N, 1, 28, 28) --> (N, 1*28*28)"""
    batch_size = len(batch_list)
    X = torch.stack([item[0] for item in batch_list], dim=0).reshape(batch_size, -1)
    y = torch.LongTensor([item[1] for item in batch_list])
    return X, y
