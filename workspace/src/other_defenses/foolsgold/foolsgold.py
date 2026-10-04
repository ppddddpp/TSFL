"""
This file is derived from the FedDLAD's implementation for FoolsGold.
Minor modifications have been made to adapt it to the PyTorch state_dict format
and the class-based pipeline of this project.

[1] C. Fung, C. J. M. Yoon, and I. Beschastnikh
    "The Limitations of Federated Learning in Sybil Settings," 
    in Proceedings of the 23rd International Symposium on Research in Attacks, Intrusions and Defenses (RAID), 2020.

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
import sklearn.metrics.pairwise as smp
import copy

class FoolsGold:
    def __init__(self):
        self.history_updates = {}
        
        self.raw_weights = {}
        self.client_weights = {}
        self.aggregation_coefficients = {}
        self.filtered_client_ids = []
        self.contributing_client_ids = []
        self.selected_indices = []

    def run(self, global_model, client_updates_dict):
        self.raw_weights = {}
        self.client_weights = {}
        self.aggregation_coefficients = {}
        self.filtered_client_ids = []
        self.contributing_client_ids = []
        self.selected_indices = []

        agent_ids = list(client_updates_dict.keys())
        n_clients = len(agent_ids)

        if not client_updates_dict:
            return copy.deepcopy(global_model)
        
        parameter_items = list(global_model.named_parameters())
        bn_keys = [
            name for name, _ in global_model.named_buffers()
            if name.endswith("running_mean") or name.endswith("running_var")
        ]

        for agent_id, update in client_updates_dict.items():
            tensors_to_cat = []
            for key, parameter in parameter_items:
                if key not in update:
                    raise KeyError(f"Client {agent_id} is missing parameter '{key}'.")
                
                client_tensor = update[key]
                if client_tensor.shape != parameter.shape:
                    raise ValueError(f"Client {agent_id}, key '{key}': expected {tuple(parameter.shape)}, got {tuple(client_tensor.shape)}.")
                
                flat_tensor = client_tensor.detach().reshape(-1).float().to(parameter.device)
                
                if not torch.isfinite(flat_tensor).all():
                    raise RuntimeError(f"Client {agent_id} contains NaN or Inf in parameter '{key}'.")
                
                tensors_to_cat.append(flat_tensor)
            
            flat_vec = torch.cat(tensors_to_cat).cpu().numpy()
            
            if agent_id not in self.history_updates:
                self.history_updates[agent_id] = flat_vec.copy()
            else:
                self.history_updates[agent_id] += flat_vec

            if not np.isfinite(self.history_updates[agent_id]).all():
                raise RuntimeError(f"FoolsGold history for client {agent_id} contains NaN or Inf.")

        updates = np.stack([self.history_updates[agent_id] for agent_id in agent_ids])
        
        if n_clients == 1:
            # Edge case: If there's only one client, we can't compute cosine similarity. 
            # Assign it a default weight of 1 for faster convergence.
            wv = np.ones(1, dtype=np.float64)
        else:
            cs = smp.cosine_similarity(updates) - np.eye(n_clients)
            maxcs = np.max(cs, axis=1)

            # Pardoning
            for i in range(n_clients):
                for j in range(n_clients):
                    if i == j:
                        continue
                    # Prevent ZeroDivisionError in extreme cases
                    if maxcs[i] < maxcs[j] and maxcs[j] != 0:
                        cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

            wv = 1 - (np.max(cs, axis=1))
            wv[wv > 1] = 1
            wv[wv < 0] = 0

            max_wv = np.max(wv)
            if max_wv > 0:
                wv = wv / max_wv
            wv[(wv == 1)] = 0.99

            with np.errstate(divide="ignore", invalid="ignore"):
                wv = np.log(wv / (1.0 - wv)) + 0.5

            wv = np.nan_to_num(
                wv,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            wv = np.clip(wv, 0.0, 1.0)

        # Unflatten logic to correctly update the global model.
        raw_weight_sum = float(np.sum(wv))

        self.raw_weights = {cid: float(wv[i]) for i, cid in enumerate(agent_ids)}
        self.aggregation_coefficients = {cid: float(wv[i]) / n_clients for i, cid in enumerate(agent_ids)}

        if raw_weight_sum <= 1e-12:
            print("[FOOLSGOLD WARNING] All client weights are zero; skipping the aggregation round.")
            self.client_weights = {cid: 0.0 for cid in agent_ids}
            return copy.deepcopy(global_model)

        self.client_weights = {
            cid: float(wv[i]) / raw_weight_sum 
            for i, cid in enumerate(agent_ids)
        }

        self.contributing_client_ids = [cid for cid in agent_ids if self.raw_weights[cid] > 0.0]
        
        uniform_weight = 1.0 / n_clients
        self.filtered_client_ids = [cid for cid in agent_ids if self.client_weights[cid] > uniform_weight]
        self.selected_indices = [index for index, cid in enumerate(agent_ids) if self.client_weights[cid] > uniform_weight]

        # Update Global Model
        new_global_model = copy.deepcopy(global_model)
        new_global_state = new_global_model.state_dict()
        
        aggregated_update = {k: torch.zeros_like(parameter, dtype=torch.float32) for k, parameter in parameter_items}
        for i, agent_id in enumerate(agent_ids):
            weight = self.aggregation_coefficients[agent_id]
            client_update = client_updates_dict[agent_id]
            for key, parameter in parameter_items:
                aggregated_update[key] += client_update[key].to(aggregated_update[key].device) * weight
                
        for key, parameter in parameter_items:
            target_device = new_global_state[key].device
            if new_global_state[key].dtype != aggregated_update[key].dtype:
                aggregated_layer = aggregated_update[key].to(new_global_state[key].dtype)
            else:
                aggregated_layer = aggregated_update[key]
            new_global_state[key] += aggregated_layer.to(target_device)
            
        for key in bn_keys:
            bn_update = torch.zeros_like(new_global_state[key], dtype=torch.float32)
            for cid in agent_ids:
                if key not in client_updates_dict[cid]:
                    continue
                    
                client_bn_update = client_updates_dict[cid][key]
                if client_bn_update.shape != new_global_state[key].shape:
                    raise ValueError(f"Client {cid}, buffer '{key}': expected {tuple(new_global_state[key].shape)}, got {tuple(client_bn_update.shape)}.")
                
                bn_update += (
                    client_bn_update.detach().to(device=bn_update.device, dtype=torch.float32) 
                    * self.aggregation_coefficients[cid]
                )
            new_global_state[key] += bn_update.to(dtype=new_global_state[key].dtype)
            
        new_global_model.load_state_dict(new_global_state)
        return new_global_model