"""
This file is derived from the FedDLAD's implementation.
Minor modifications have been made to adapt our research pipeline
and the class-based pipeline of this project.

[2] Ding, Binbin and Yang, Penghui and Huang, Sheng-Jun
    "FedDLAD: A Federated Learning Dual-Layer Anomaly Detection Framework for Enhancing Resilience Against Backdoor Attacks"
    in Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}

Concept source code taken from:
https://github.com/dingbinb/FedDLAD

Thanks for their contribution!

Modified by ppdddd et al.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, Subset

def add_trigger(x, dataset='cifar10', trigger='CBA', agent_idx=-1):
    # x is expected to be shape (C, H, W)
    if dataset in ['cifar10', 'svhn']:
        if trigger == 'DBA':
            if agent_idx == -1:
                for d in range(0, 3):
                    for i in range(25, 30):
                        x[d, i, 27] = 255
                    for j in range(25, 30):
                        x[d, 27, j] = 255
            else:
                if agent_idx % 4 == 0:
                    for d in range(0, 3):
                        for i in range(25, 28):
                            x[d, i, 27] = 255
                elif agent_idx % 4 == 1:
                    for d in range(0, 3):
                        for i in range(27, 30):
                            x[d, i, 27] = 255
                elif agent_idx % 4 == 2:
                    for d in range(0, 3):
                        for j in range(25, 28):
                            x[d, 27, j] = 255
                elif agent_idx % 4 == 3:
                    for d in range(0, 3):
                        for j in range(27, 30):
                            x[d, 27, j] = 255

        elif trigger == 'CBA':
            for d in range(0, 3):
                for i in range(26, 30):
                    for j in range(26, 30):
                        x[d, i, j] = 255

    elif dataset in ['mnist', 'fmnist']:
        # For grayscale, C=1, so we index x[0, i, j]
        if trigger == 'DBA':
            if agent_idx == -1:
                for i in range(23, 28):
                    x[0, i, 25] = 255
                for j in range(23, 28):
                    x[0, 25, j] = 255
            else:
                if agent_idx % 2 == 0:
                    for i in range(23, 28):
                        x[0, i, 25] = 255
                elif agent_idx % 2 == 1:
                    for j in range(23, 28):
                        x[0, 25, j] = 255

        elif trigger == 'CBA':
            for i in range(23, 28):
                for j in range(23, 28):
                    x[0, i, j] = 255
    return x

class PoisonedClientDataset(Dataset):
    def __init__(self, original_subset: Subset, dataset_name: str, attack_type: str, agent_idx: int, base_class=0, target_class=5, poison_ratio=0.5):
        self.original_subset = original_subset
        self.dataset_name = dataset_name.lower().strip()
        self.attack_type = attack_type.upper()
        self.agent_idx = agent_idx
        self.base_class = base_class
        self.target_class = target_class
        self.poison_ratio = poison_ratio
        
        if self.dataset_name == "cifar10":
            self.mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
            self.std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
        elif self.dataset_name == "cifar100":
            self.mean = np.array([0.5071, 0.4867, 0.4408]).reshape(3, 1, 1)
            self.std = np.array([0.2675, 0.2565, 0.2761]).reshape(3, 1, 1)
        else:
            self.mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
            self.std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)

    def __len__(self):
        return len(self.original_subset)

    def __getitem__(self, idx):
        img, label = self.original_subset[idx]
        
        if label == self.base_class and (idx % int(1/self.poison_ratio) == 0):
            img_np = img.clone().cpu().numpy()
            
            img_np = (img_np * self.std) + self.mean
            img_np = np.clip(img_np, 0.0, 1.0)
            img_np = (img_np * 255).astype(np.uint8)

            img_np = add_trigger(img_np, dataset=self.dataset_name, trigger=self.attack_type, agent_idx=self.agent_idx)
            
            img_np = img_np.astype(np.float32) / 255.0
            img_np = (img_np - self.mean) / self.std
                
            img = torch.tensor(img_np)
            label = self.target_class

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return img, label