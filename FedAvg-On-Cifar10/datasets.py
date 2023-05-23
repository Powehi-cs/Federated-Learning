import logging
import sys

import torch
from torchvision import datasets, transforms


def get_dataset(dir: str, name: str):
    if name == 'mnist':
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
    elif name == 'cifar':
        transforms_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transforms_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transforms_test)
    else:
        logging.error("data name error")
        sys.exit(1)

    return train_dataset, eval_dataset
