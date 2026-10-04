import numpy as np
from torch.utils.data import Dataset, Subset
from typing import Dict

from helpers import log_and_print

class DirichletDataPartitioner:
    def __init__(self, dataset: Dataset, n_clients: int, alpha: float = 0.5, seed: int = 2709):
        """
        Splits a PyTorch dataset into Non-IID partitions using a Dirichlet distribution.
        This is the absolute standard benchmark for testing Federated Learning robustness.

        Args:
            dataset (Dataset): The global training dataset (e.g., CIFAR-10).
            n_clients (int): Total number of clients in the simulation.
            alpha (float): The concentration parameter. 
                            - alpha = 100: Almost perfectly IID (balanced).
                            - alpha = 0.5: Standard Non-IID (messy, realistic).
                            - alpha = 0.1: Extreme Non-IID (clients only have 1 or 2 classes).
            seed (int): For reproducible paper results.
        """
        self.dataset = dataset
        self.n_clients = n_clients
        self.alpha = alpha
        self.seed = seed

    def generate_client_datasets(self, log_file_path: str=None) -> Dict[str, Subset]:
        """
        Returns a dictionary mapping 'client_id' to their specific PyTorch Subset.
        """
        log_and_print(f"Generating Non-IID client datasets with alpha={self.alpha} for {self.n_clients} clients...", log_file_path=log_file_path)
        np.random.seed(self.seed)
        
        # Extract all labels from the dataset to know the class distribution
        try:
            labels = np.array(self.dataset.targets) # Works for CIFAR/MNIST
        except AttributeError:
            labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
            
        n_classes = len(np.unique(labels))
        
        # Prepare an empty list of indices for each client
        client_indices = {f"client_{i}": [] for i in range(self.n_clients)}
        
        # Distribute each class to clients based on Dirichlet proportions
        for k in range(n_classes):
            # Find all images belonging to class 'k'
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            
            # Generate Dirichlet proportions for this class across all clients
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.n_clients))
            
            # Convert proportions into actual number of samples
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # Split the indices for this class and assign to clients
            idx_k_split = np.split(idx_k, proportions)
            
            for i, client_id in enumerate(client_indices.keys()):
                client_indices[client_id].extend(idx_k_split[i].tolist())
                
        # Convert lists of indices into PyTorch Subset objects
        client_datasets = {}
        for client_id, indices in client_indices.items():
            # Shuffle the client's local dataset so they don't train on all "cats" then all "dogs"
            np.random.shuffle(indices)
            client_datasets[client_id] = Subset(self.dataset, indices)
            
        return client_datasets