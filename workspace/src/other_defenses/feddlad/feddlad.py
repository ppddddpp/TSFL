"""
This file is derived from the FedDLAD's implementation.
Minor modifications have been made to adapt it to the PyTorch state_dict format
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
from sklearn.metrics.pairwise import cosine_similarity
from pyod.models.cof import COF
import copy

class FedDLAD:
    def __init__(self, bg: int, pg: int, iqr_scale: float = 0.6, device='cuda'):
        # [MODIFIED] Replaced undefined 'self.args' with instance variables
        self.bg = max(1, bg)  
        self.pg = max(1, pg)  
        self.iqr_scale = iqr_scale
        self.device = device
        self.selected_indices = []
        self.selected_client_ids = []
        self.effective_weights = {} 

    def run(self, global_model, client_updates_dict, client_data_sizes=None):
        self.selected_indices = []
        self.selected_client_ids = []
        self.effective_weights = {}

        if not client_updates_dict:
            return copy.deepcopy(global_model)
        
        if client_data_sizes is None:
            self.agent_data_sizes = {cid: 1.0 for cid in client_updates_dict}
        else:
            self.agent_data_sizes = {
                cid: float(client_data_sizes.get(cid, 1.0))
                for cid in client_updates_dict
            }

        parameter_items = list(global_model.named_parameters())
        feature_keys = [name for name, _ in parameter_items]
        
        bn_keys = [
            name for name, _ in global_model.named_buffers()
            if name.endswith("running_mean") or name.endswith("running_var")
        ]
        
        safe_updates = {}
        cur_parameters_dict = {}
        
        for agent_id, update_state in client_updates_dict.items():
            update_parts = []
            current_parameter_parts = []

            for name, global_parameter in parameter_items:
                if name not in update_state:
                    raise KeyError(f"Client {agent_id} missing parameter: {name}")

                delta = update_state[name].detach().to(
                    device=self.device,
                    dtype=global_parameter.dtype
                )

                if delta.shape != global_parameter.shape:
                    raise ValueError(f"Shape mismatch at {name}: got {delta.shape}, expected {global_parameter.shape}")

                update_parts.append(delta.flatten())

                local_parameter = global_parameter.detach().to(self.device) + delta
                current_parameter_parts.append(local_parameter.flatten())

            safe_updates[agent_id] = torch.cat(update_parts)
            cur_parameters_dict[agent_id] = torch.cat(current_parameter_parts)
        
        total_update_flat = self.combined_aggregation(safe_updates, cur_parameters_dict)
        
        new_global_model = copy.deepcopy(global_model)
        
        start_idx = 0
        with torch.no_grad():
            for name, parameter in new_global_model.named_parameters():
                num_elements = parameter.numel()
                layer_update = total_update_flat[start_idx : start_idx + num_elements].view_as(parameter)
                
                parameter.add_(
                    layer_update.to(
                        device=parameter.device,
                        dtype=parameter.dtype
                    )
                )
                start_idx += num_elements

        if start_idx != total_update_flat.numel():
            raise RuntimeError("Update vector and model mismatch.")
            
        new_global_state = new_global_model.state_dict()
        if bn_keys and self.effective_weights:
            for key in bn_keys:
                bn_update = torch.zeros_like(new_global_state[key], dtype=torch.float32)
                for cid, weight in self.effective_weights.items():
                    if key not in client_updates_dict[cid]:
                        continue
                        
                    client_buffer = client_updates_dict[cid][key]
                    if client_buffer.shape != new_global_state[key].shape:
                        raise ValueError(f"Client {cid}, buffer '{key}' mismatch.")
                        
                    bn_update += (
                        client_buffer.detach().to(device=bn_update.device, dtype=torch.float32) * weight
                    )
                new_global_state[key] += bn_update.to(new_global_state[key].dtype)

        new_global_model.load_state_dict(new_global_state)

        ordered_ids = list(client_updates_dict.keys())
        self.selected_indices = [
            ordered_ids.index(cid) 
            for cid in self.selected_client_ids
            if cid in ordered_ids
        ]

        return new_global_model

    def COF(self, agent_cur_parameters_dict):
        agent_ids = list(agent_cur_parameters_dict.keys())
        n_samples = len(agent_ids)
        if n_samples == 0:
            return []
        if n_samples == 1:
            return agent_ids
        
        parameter_vectors = {agent_id: update.detach().cpu().numpy().astype(np.float64)
                             for agent_id, update in agent_cur_parameters_dict.items()}
        parameter_matrix = np.array(list(parameter_vectors.values()))

        cosine_distance = 1 - cosine_similarity(parameter_matrix)
        
        n_neighbors = min(24, n_samples - 1)
        
        cof = COF(contamination=0.1, n_neighbors=n_neighbors) # [MODIFIED] Applied safe n_neighbors
        cof.fit(cosine_distance)
        connectivity_distances = cof.decision_function(cosine_distance)

        agent_anomaly_scores = {agent_id: connectivity_distances[idx] for idx, agent_id in
                                enumerate(parameter_vectors.keys())}

        k1 = self.bg # [MODIFIED] Replaced self.args.bg with self.bg
        sorted_id = sorted(agent_anomaly_scores.items(), key=lambda x: x[1])
        benign_id = sorted_id[:k1]
        benign_id_list = [item[0] for item in benign_id]
        return benign_id_list

    def combined_aggregation(self, agent_updates_dict, agent_cur_parameters_dict):
    
        norms = []
        reference_ids = self.COF(agent_cur_parameters_dict)

        for agent_id, update in agent_updates_dict.items():
            if agent_id in reference_ids:
                norm = np.linalg.norm(update.detach().cpu().numpy())
                norms.append(norm)
        
        median_norm = float(np.median(norms)) if norms else 0.0
        if not np.isfinite(median_norm):
            raise RuntimeError("median_norm is not finite.")

        for agent_id, update in agent_updates_dict.items():
            update_norm = np.linalg.norm(update.detach().cpu().numpy())
            if update_norm > median_norm:
                scale_factor = median_norm / update_norm
                update *= scale_factor
                agent_updates_dict[agent_id] = update

        all_updates = []
        total_outlier_count = 0
        for agent_id, update in agent_updates_dict.items():
            update_vector = update.clone()
            all_updates.append(update_vector.cpu().numpy())
        all_updates = np.array(all_updates)
        
        q1 = np.percentile(all_updates, 25, axis=0)
        q3 = np.percentile(all_updates, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_scale * iqr
        upper_bound = q3 + self.iqr_scale * iqr

        dimension_outlier_count = 0
        for agent_id, update in agent_updates_dict.items():
            update_vector = update.clone().cpu().numpy()
            mask = (update_vector < lower_bound) | (update_vector > upper_bound)
            total_outlier_count += np.sum(mask)
            update_vector[mask] = -update_vector[mask]
            dimension_outlier_count += mask
            
            # [MODIFIED] Replaced self.args.device with self.device
            agent_updates_dict[agent_id] = torch.from_numpy(update_vector).float().to(self.device)

        reference_update, reference_data = 0, 0
        for agent_id, update in agent_updates_dict.items():
            if agent_id in reference_ids:
                reference_data += self.agent_data_sizes[agent_id]
                reference_update += self.agent_data_sizes[agent_id] * agent_updates_dict[agent_id]
        
        if reference_data > 0: # [MODIFIED] Safe division
            reference_update /= reference_data
        else:
            reference_update = torch.zeros_like(agent_updates_dict[list(agent_updates_dict.keys())[0]])

        pardoned_ids, score_dict = self.secondary_filtering(reference_ids, reference_update, agent_updates_dict,
                                                            k2=self.pg) # [MODIFIED] Replaced self.args.pg

        self.selected_client_ids = list(reference_ids) + list(pardoned_ids)

        selected_count = len(reference_ids) + len(pardoned_ids)
        self.effective_weights = {}

        if selected_count > 0:
            # Effective coefficients of reference clients.
            reference_group_weight = (
                len(reference_ids) / selected_count
            )

            if reference_data > 0:
                for cid in reference_ids:
                    self.effective_weights[cid] = (
                        reference_group_weight
                        * self.agent_data_sizes[cid]
                        / reference_data
                    )

            # Effective coefficients of pardoned clients.
            if pardoned_ids:
                pardoned_score_sum = sum(
                    score_dict[cid]
                    for cid in pardoned_ids
                )

                pardoned_group_weight = (
                    len(pardoned_ids) / selected_count
                )

                if pardoned_score_sum > 0:
                    for cid in pardoned_ids:
                        self.effective_weights[cid] = (
                            pardoned_group_weight
                            * score_dict[cid]
                            / pardoned_score_sum
                        )
                        
        pardoned_update, score = 0, 0

        if pardoned_ids:
            for agent_id in pardoned_ids:
                pardoned_update += score_dict[agent_id] * agent_updates_dict[agent_id]
                score += score_dict[agent_id]
            pardoned_update /= score
            total_update = len(reference_ids) / len(reference_ids + pardoned_ids) * reference_update + len(
                pardoned_ids) / len(reference_ids + pardoned_ids) * pardoned_update
        else:
            total_update = reference_update
            
        # self.print(reference_ids, pardoned_ids) # [MODIFIED] Commented out to prevent terminal spam
        return total_update
    
    def print(self, reference_ids, pardoned_ids):
        print(f'PardonedID---{pardoned_ids}')
        print(f'TrustedID---{reference_ids}')
        print(f'IDs---{reference_ids + pardoned_ids}')

    def secondary_filtering(self, ref_ids, reference_update, agent_updates_dict, k2):
        score_dict = {}
        for agent_id, update in agent_updates_dict.items():
            if agent_id not in ref_ids:
                sim_cosine = \
                cosine_similarity(reference_update.cpu().numpy().reshape(1, -1), update.cpu().numpy().reshape(1, -1))[
                    0][0]
                
                # [MODIFIED] Replaced undefined 'self.relu' with standard math.max
                score = max(0.0, float(sim_cosine)) 
                score_dict[agent_id] = score

        filtered_score_dict = {agent_id: score for agent_id, score in score_dict.items() if score != 0}
        sorted_ids = sorted(filtered_score_dict, key=filtered_score_dict.get, reverse=True)[:k2]
        return sorted_ids, score_dict