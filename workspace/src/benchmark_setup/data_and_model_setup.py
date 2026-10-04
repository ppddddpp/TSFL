import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path

from models.resnet import ResNet18

class DataModelSetup:
    """
    Class to generate datasets with transforms and their corresponding models.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def get_dataset(self, dataset_name: str):
        """Returns the (train_dataset, test_dataset) with preprocessing."""
        dataset_name = dataset_name.lower()
        
        if dataset_name == "cifar10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_set = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=transform_train)
            test_set = datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=transform_test)
            
        elif dataset_name == "cifar100":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])
            train_set = datasets.CIFAR100(root=self.data_dir, train=True, download=True, transform=transform_train)
            test_set = datasets.CIFAR100(root=self.data_dir, train=False, download=True, transform=transform_test)

        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported yet.")

        return train_set, test_set

    def get_model(self, dataset_name: str, device: torch.device) -> nn.Module:
        """Returns the exact global model architecture for the requested dataset."""
        dataset_name = dataset_name.lower()
        
        if dataset_name == "cifar10":
            model = ResNet18(num_classes=10)
        elif dataset_name == "cifar100":
            model = ResNet18(num_classes=100)
        else:
            raise ValueError(f"Model for dataset '{dataset_name}' is not configured.")
            
        return model.to(device)