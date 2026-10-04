"""
This file is adapted for this project based on the algorithmic ideas of FLTrust.

The FLTrust defense strategy and trust bootstrapping concept are introduced in:
[1] X. Cao, M. Fang, J. Liu, and N. Z. Gong,
    "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping,"
    in Proc. NDSS, 2021.

[2] Ding, Binbin and Yang, Penghui and Huang, Sheng-Jun
    "FedDLAD: A Federated Learning Dual-Layer Anomaly Detection Framework for Enhancing Resilience Against Backdoor Attacks"
    in Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}

Concept source code taken from:
https://github.com/dingbinb/FedDLAD

Thanks for their contribution!

Modified by ppdddd et al.
"""

import torch
import torch.nn.functional as F
import copy

class FLTrustDefense:
    def __init__(self, server_proxy_loader, device, global_lr=1.0):
        self.server_proxy_loader = server_proxy_loader
        self.device = device
        self.global_lr = global_lr
        
        self.filtered_client_ids = []
        self.selected_indices = []
        self.client_weights = {}
        self.contributing_client_ids = []

    def compute_server_update(self, global_model, current_round):
        """Compute the trusted root update (Delta W0) using client's LR schedule."""
        server_model = copy.deepcopy(global_model).to(self.device)
        server_model.train()

        current_lr = 0.1 * (0.99 ** (current_round - 1))

        optimizer = torch.optim.SGD(
            server_model.parameters(),
            lr=current_lr,
            momentum=0.9,
        )
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(3):
            for images, labels in self.server_proxy_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = server_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        global_parameters = dict(global_model.named_parameters())
        server_parameters = dict(server_model.named_parameters())

        server_update = {
            name: (
                server_parameters[name].detach()
                - global_parameters[name].detach().to(self.device)
            )
            for name in global_parameters
        }

        return server_update

    def flatten_update(self, update_dict, reference_keys):
        """ Helper to convert dictionary updates into 1D tensors for cosine similarity. """
        vectors = [update_dict[k].flatten() for k in reference_keys if "num_batches_tracked" not in k]
        return torch.cat(vectors)

    def run(self, global_model, client_updates_dict, current_round):
        self.contributing_client_ids = []
        self.filtered_client_ids = []
        self.selected_indices = []
        self.client_weights = {}

        if not client_updates_dict or len(client_updates_dict) == 0:
            return copy.deepcopy(global_model)

        parameter_items = list(global_model.named_parameters())
        feature_keys = [name for name, _ in parameter_items]
        
        bn_keys = [
            name for name, _ in global_model.named_buffers()
            if name.endswith("running_mean") or name.endswith("running_var")
        ]
        
        safe_client_updates_dict = {}
        for agent_id, update in client_updates_dict.items():
            safe_update = {}
            for key, parameter in parameter_items:
                if key not in update:
                    raise KeyError(f"Client {agent_id} is missing '{key}'.")
                
                client_tensor = update[key]
                if client_tensor.shape != parameter.shape:
                    raise ValueError(f"Client {agent_id}, key '{key}': expected {tuple(parameter.shape)}, got {tuple(client_tensor.shape)}.")
                
                flat_tensor = client_tensor.detach().reshape(-1).float().to(parameter.device)
                if not torch.isfinite(flat_tensor).all():
                    raise RuntimeError(f"Client {agent_id} contains NaN or Inf in parameter '{key}'.")

                safe_update[key] = client_tensor.detach().to(device=parameter.device, dtype=parameter.dtype)
            safe_client_updates_dict[agent_id] = safe_update

        # Calculate reference_update (g0) and its norm
        server_update_dict = self.compute_server_update(global_model, current_round)
        reference_update = self.flatten_update(server_update_dict, feature_keys)
        reference_update_norm = torch.norm(reference_update)

        if (
            not torch.isfinite(reference_update).all()
            or not torch.isfinite(reference_update_norm)
            or reference_update_norm <= 1e-12
        ):
            print("[FLTRUST WARNING] Invalid or zero server root update; skipping this round.")
            self.client_weights = {cid: 0.0 for cid in client_updates_dict}
            return copy.deepcopy(global_model)

        total_score = 0.0
        # Initialize total_update as dictionary instead of 1D tensor for framework compatibility
        total_update_dict = {k: torch.zeros_like(parameter) for k, parameter in parameter_items}
                
        trust_scores = {}
        cosine_scores = {}
        client_norms = []
        scale_factors = []

        for agent_id, update_dict in safe_client_updates_dict.items():
            agent_update = self.flatten_update(update_dict, feature_keys)
            agent_update_norm = torch.norm(agent_update).item()
            client_norms.append(agent_update_norm)

            # Author's code logic: if agent_update_norm != 0:
            if agent_update_norm > 1e-9:
                # Author's code: scale_factor = reference_update_norm / agent_update_norm
                scale_factor = (reference_update_norm / agent_update_norm).item()
                scale_factors.append(scale_factor)

                # Author's code: sim_cosine = torch.nn.functional.cosine_similarity(...)
                sim_cosine = F.cosine_similarity(reference_update.unsqueeze(0), agent_update.unsqueeze(0)).item()

                cosine_scores[agent_id] = sim_cosine

                # Author's code: trust_score = self.relu(sim_cosine...)
                trust_score = max(0.0, sim_cosine)
                
                # Author's code: total_score += score_tensor
                trust_scores[agent_id] = trust_score
                total_score += trust_score

                # Author's code: update = agent_updates_dict[agent_id] * scale_factor
                # Author's code: weighted_update = score_tensor * update
                # Author's code: total_update += weighted_update
                for k in feature_keys:
                    scaled_layer = update_dict[k] * scale_factor
                    weighted_layer = scaled_layer * trust_score
                    total_update_dict[k] += weighted_layer
            else:
                trust_scores[agent_id] = 0.0
                cosine_scores[agent_id] = 0.0
                scale_factors.append(0.0)

        num_clients = len(client_updates_dict)
        positive_count = sum(score > 0.0 for score in trust_scores.values())
        cosine_values = list(cosine_scores.values())

        client_norm_mean = sum(client_norms) / len(client_norms) if client_norms else 0.0
        scale_mean = sum(scale_factors) / len(scale_factors) if scale_factors else 0.0
        cos_mean = (sum(cosine_values) / len(cosine_values)) if cosine_values else 0.0


        print(
            f"   [FLTrust Diagnostics] root_norm={reference_update_norm.item():.4e} | "
            f"client_norm_mean={client_norm_mean:.4e} | scale_mean={scale_mean:.4f} | "
            f"positive_trust={positive_count}/{num_clients} | "
            f"cos_mean={cos_mean:.4f} | "
            f"cos_min={min(cosine_values) if cosine_values else 0:.4f} | "
            f"cos_max={max(cosine_values) if cosine_values else 0:.4f}"
        )

        if total_score < 1e-12:
            print("[FLTRUST WARNING] No client received positive trust; skipping this aggregation round.")
            self.client_weights = {cid: 0.0 for cid in client_updates_dict}
            return copy.deepcopy(global_model)

        uniform_weight = 1.0 / num_clients
        
        self.client_weights = {
            cid: trust_scores.get(cid, 0.0) / total_score
            for cid in client_updates_dict
        }

        self.contributing_client_ids = [
            cid for cid, score in trust_scores.items() if score > 1e-12
        ]

        self.filtered_client_ids = [
            cid for cid, weight in self.client_weights.items() if weight > uniform_weight
        ]
        
        ordered_ids = list(client_updates_dict.keys())
        self.selected_indices = [
            index for index, cid in enumerate(ordered_ids) if self.client_weights.get(cid, 0.0) > uniform_weight
        ]

        # Update Model
        new_global_model = copy.deepcopy(global_model)
        with torch.no_grad():
            state = new_global_model.state_dict()

            for k in feature_keys:
                averaged_layer = total_update_dict[k] / total_score
                averaged_layer = averaged_layer * self.global_lr
                
                if state[k].dtype != averaged_layer.dtype:
                    averaged_layer = averaged_layer.to(state[k].dtype)
                state[k] += averaged_layer

            for key in bn_keys:
                bn_update = torch.zeros_like(state[key], dtype=torch.float32)
                for cid, weight in self.client_weights.items():
                    if key in client_updates_dict[cid]:
                        bn_update += (client_updates_dict[cid][key].to(bn_update.device, dtype=torch.float32) * weight)
                
                bn_update = bn_update * self.global_lr
                state[key] += bn_update.to(state[key].dtype)
                
            new_global_model.load_state_dict(state)

        return new_global_model