from copy import deepcopy
from typing import Optional, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class DatasetEvaluator:
    def __init__(
            self,
            dataset_loader: DataLoader,
            max_samples: int = 512,
            T_kl: float = 2.0,
            seed: int = 2709,
            gamma_sen: int = 5
        ):
        self.T_kl = T_kl
        self.seed = seed
        self.classwise_kl_upper_threshold = 10.0  # Classwise KL upper threshold
        self.gamma_sen = gamma_sen
        
        # Build a small, balanced subset of the data for fast evaluation
        subset = self._get_stratified_dataset_subset(dataset_loader, max_samples=max_samples, seed=self.seed)
        self.dataset_subset = self._build_dataset_loader_from_subset(subset)
        self.dataset_labels = self._extract_labels(self.dataset_subset)

        # Cache
        self.param_keys_cache = None

    def evaluate_on_dataset(
            self, 
            global_model: nn.Module, 
            client_updates: Dict[str, dict], 
            device: Optional[torch.device] = None
        ) -> dict:
        if device is None:
            try:
                device = next(global_model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        # Evaluate the global model
        logits_g, baseline_acc, baseline_loss = self._forward_collect(global_model, self.dataset_subset, device)

        model_clone = self._clone_model(global_model).to(device)
        global_sd_original = global_model.state_dict()
        ids = list(client_updates.keys())
        client_scores = {}
        warnings_log = {}

        # Evaluate each client update individually
        for client_id in ids:
            client_delta = client_updates[client_id]
            
            # Load the clone with the global model
            model_clone.load_state_dict(global_sd_original) 
            sd_to_update = model_clone.state_dict()

            # Apply delta 
            warnings = self._safe_apply_delta(model_clone, sd_to_update, client_delta)
            warnings_log[client_id] = warnings

            # Evaluate the updated clone
            logits_p, new_acc, new_loss = self._forward_collect(model_clone, self.dataset_subset, device)

            # Calculate metrics
            delta_acc = float(new_acc - baseline_acc)
            delta_loss = float(new_loss - baseline_loss)
            kl_div = self._compute_kl(logits_g, logits_p)
            classwise_kl = self._compute_classwise_kl(logits_g, logits_p, self.dataset_labels)

            # Build standardized client score dictionary
            client_scores[client_id] = {
                "reference_delta_acc_norm": np.tanh(delta_acc * self.gamma_sen), # Negative is bad (accuracy dropped)
                "reference_delta_loss_norm": np.tanh(delta_loss), # Positive is bad (loss increased)
                "reference_kl_norm": np.tanh(kl_div),
                "reference_classwise_kl_norm": np.tanh(classwise_kl)
            }

        return {
            "keep_ids": ids,
            "client_scores": client_scores,
            
            # Additional contextual info
            "baseline_acc": float(baseline_acc),
            "baseline_loss": float(baseline_loss),
            "warnings": warnings_log
        }

    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        return deepcopy(model)

    def _safe_apply_delta(self, model, state_dict, client_delta):
        warnings = []
        # Only consider keys that are in the model's state_dict and are parameters (not buffers)
        if self.param_keys_cache is None:
            self.param_keys_cache = set(dict(model.named_parameters()).keys())
        param_keys = self.param_keys_cache

        for k, v in client_delta.items():
            if k not in state_dict:
                warnings.append(f"missing_key:{k}")
                continue

            if k not in param_keys:
                # Skip buffers (BN stats, etc.)
                warnings.append(f"skipped_buffer:{k}")
                continue

            if state_dict[k].shape != v.shape:
                warnings.append(f"shape_mismatch:{k}")
                continue

            state_dict[k].add_(
                v.to(
                    device=state_dict[k].device,
                    dtype=state_dict[k].dtype
                )
            )

        return warnings

    @torch.no_grad()
    def _forward_collect(self, model, loader, device):
        model.eval().to(device)
        logits_list, preds_list, targets_list = [], [], []
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0

        for batch in loader:
            if len(batch) == 3:
                # Some input style is (input IDs, attention mask, labels)
                x, mask, y = batch
            else:
                x, y = batch
                mask = None

            x, y = x.to(device), y.to(device)
            out = model(x, attention_mask=mask) if mask is not None else model(x)

            loss = criterion(out, y)
            total_loss += loss.item() * len(x)
            total_samples += len(x)

            logits_list.append(out.detach().cpu())
            preds_list.append(out.argmax(dim=1).cpu())
            targets_list.append(y.cpu())

        logits = torch.cat(logits_list)
        preds = torch.cat(preds_list)
        targets = torch.cat(targets_list)

        acc = (preds == targets).float().mean().item()
        avg_loss = total_loss / total_samples

        return logits, acc, avg_loss

    def _compute_kl(self, logits_g, logits_p):
        # Target (global) uses regular softmax (linear probabilities)
        p = torch.softmax(logits_g / self.T_kl, dim=1)
        # Input (local uses log_softmax (log probabilities)
        log_q = torch.log_softmax(logits_p / self.T_kl, dim=1)
        
        # PyTorch natively protects against p=0 when log_target=False (default)
        kl = F.kl_div(log_q, p, reduction='none').sum(dim=1)
        return float(torch.clamp(kl, 0, 10).mean().item())
    
    def _compute_classwise_kl(self, logits_g, logits_p, labels):
        # Target (global) uses regular softmax (linear probabilities)
        p = torch.softmax(logits_g / self.T_kl, dim=1)
        # Input (local) uses log_softmax (log probabilities)
        log_q = torch.log_softmax(logits_p / self.T_kl, dim=1)
        
        # PyTorch natively protects against p=0 when log_target=False (default)
        kl = F.kl_div(log_q, p, reduction='none').sum(dim=1)

        # Compute KL for each class
        class_kl = []
        for c in torch.unique(labels):
            mask = labels == c
            if mask.sum() > 0:
                class_kl.append(kl[mask].mean().item())

        if not class_kl:
            return 0.0
        
        # Penalty for one class being over the threshold
        return float(min(max(class_kl), self.classwise_kl_upper_threshold))

    @staticmethod
    def _get_stratified_dataset_subset(dataset_loader, max_samples=512, min_per_class=10, seed=2709):
        from collections import defaultdict
        import random
        random.seed(seed)
        class_buckets = defaultdict(list)

        # Loop through the dataset and collect samples for each class
        for batch in dataset_loader:
            batch = [item.cpu() if torch.is_tensor(item) else item for item in batch]
            
            if len(batch) == 3:
                x, mask, y = batch
                # Using zip to unpack
                for xi, mi, yi in zip(x, mask, y):
                    class_buckets[int(yi)].append((xi, mi, yi))
            else:
                x, y = batch
                for xi, yi in zip(x, y):
                    class_buckets[int(yi)].append((xi, yi))

        # Compute the number of samples for each class
        total_samples = sum(len(v) for v in class_buckets.values())
        counts = {}
        for c, samples in class_buckets.items():
            proportion = len(samples) / total_samples
            counts[c] = max(int(proportion * max_samples), min_per_class)

        # Get the total number of samples and adjust the number of samples for each class
        total_alloc = sum(counts.values())
        if total_alloc > max_samples:
            scale = max_samples / total_alloc
            for c in counts:
                counts[c] = max(int(counts[c] * scale), 1)

        while sum(counts.values()) > max_samples:
            largest = max(counts, key=counts.get)
            counts[largest] -= 1
        while sum(counts.values()) < max_samples:
            smallest = min(counts, key=counts.get)
            counts[smallest] += 1

        # Build the subset
        subset = []
        for c, samples in class_buckets.items():
            n_c = counts[c]
            chosen = samples if len(samples) <= n_c else random.sample(samples, n_c)
            subset.extend(chosen)

        random.shuffle(subset)
        return subset
    
    def _build_dataset_loader_from_subset(self, subset, batch_size=32):
        def collate_fn(batch):
            if len(batch[0]) == 3:
                x, mask, y = zip(*batch)
                return torch.stack(x), torch.stack(mask), torch.stack(y)
            else:
                x, y = zip(*batch)
                return torch.stack(x), torch.stack(y)
        return DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    @staticmethod
    def _extract_labels(loader):
        labels = []
        for batch in loader:
            labels.append(batch[-1].cpu()) # The last item is always `y`
        return torch.cat(labels)