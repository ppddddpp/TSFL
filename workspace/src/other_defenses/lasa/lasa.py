"""
This file is derived from Jiahao Xu @ UNR's implementation.
Minor modifications have been made for this project.

[2] J. Xu, Z. Zhang, and R. Hu,
    "Achieving Byzantine-Resilient Federated Learning via Layer-Adaptive
    Sparsified Model Aggregation," in Proc. IEEE/CVF WACV, 2025.

Original source:
https://github.com/JiiahaoXU/LASA/blob/master/algorithms/defense/lasa.py (By Jiahao Xu @ UNR)

Thanks for their contribution!

Modified by ppdddd et al.
"""

import torch
import numpy as np
import copy

from other_defenses.utils.mask_help import generate_init_mask, update_mask, apply_mask

class LASA:
    def __init__(self, num_selected_users=25, sparsity=0.3, lambda_n=1.0, lambda_s=1.0):
        self.num_selected_users = num_selected_users
        self.sparsity = sparsity
        self.lambda_n = lambda_n
        self.lambda_s = lambda_s
        
        # For evaluation / ASR logging
        self.selected_indices = []
        self.selected_indices_by_layer = {}
        self.filtered_client_ids = []

    def _parameters_dict_to_vector_flt(self, net_dict) -> torch.Tensor:
        vec = []

        # small adjust for device handling
        target_device = next(iter(net_dict.values())).device 

        for key, param in net_dict.items():
            # print(key, torch.max(param))
            # if key.split('.')[-1] == 'num_batches_tracked':
            #     continue
            vec.append(param.view(-1).to(target_device))
        return torch.cat(vec)
    
    def _vector_to_net_dict(self, vec: torch.Tensor, net_dict: dict) -> dict:
        r"""Convert one vector to the net parameters

        Args:
            vec (Tensor): a single vector represents the parameters of a model.
            parameters (Iterable[Tensor]): an iterator of Tensors that are the
                parameters of a model.
        """

        pointer = 0
        for param in net_dict.values():
            # The length of the parameter
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            param.data = vec[pointer:pointer + num_param].view_as(param).data

            # Increment the pointer
            pointer += num_param
        return net_dict

    def run(self, global_model, client_updates_dict):
        # LASA's code idea don't handle the case when there is no client updates, 
        # so we add this check to prevent potential errors.
        if not client_updates_dict or len(client_updates_dict) == 0:
            return global_model
        
        # Force all clients to have the exact same shape as the global_model
        # to prevent "inhomogeneous shape" or "stack expects equal size" errors.
        global_state = global_model.state_dict()
        reference_keys = list(global_state.keys())

        safe_client_updates_dict = {}
        for cid, update in client_updates_dict.items():
            safe_update = {}
            for k in reference_keys:
                target_device = global_state[k].device

                if k in update:
                    c_tensor = update[k]
                    expected_shape = global_state[k].shape
                    expected_len = global_state[k].numel()
                    
                    if c_tensor.shape == expected_shape:
                        safe_update[k] = c_tensor.to(target_device)
                    else:
                        # Attacker modified the tensor shape! Force it back to normal.
                        flat_t = c_tensor.flatten()
                        if flat_t.numel() > expected_len:
                            flat_t = flat_t[:expected_len]
                        elif flat_t.numel() < expected_len:
                            flat_t = torch.nn.functional.pad(flat_t, (0, expected_len - flat_t.numel()))
                        
                        # Reshape back to the 2D/3D structure expected by the layer
                        safe_update[k] = flat_t.view(expected_shape).to(target_device)
                else:
                    # Missing layer completely, fill with zeros
                    safe_update[k] = torch.zeros_like(global_state[k])
                    
            safe_client_updates_dict[cid] = safe_update
            
        # Overwrite the dirty dictionary with the perfectly aligned one
        client_updates_dict = safe_client_updates_dict

        # Handle for LASA's origin code if client_updates_dict is empty after filtering
        # Reset tracking variables for each round
        self.selected_indices = []
        self.selected_indices_by_layer = {}
        self.filtered_client_ids = []

        # Keep client ids aligned with updates
        client_ids = list(client_updates_dict.keys())
        raw_updates = list(client_updates_dict.values())

        local_updates = []
        flat_local_updates = []
        filtered_client_ids = []

        # Filter out NaN / Inf updates while preserving client ids
        for cid, update in zip(client_ids, raw_updates):
            vector = self._parameters_dict_to_vector_flt(update)

            if not torch.isfinite(vector).all():
                continue

            local_updates.append(update)
            flat_local_updates.append(vector)
            filtered_client_ids.append(cid)

        self.filtered_client_ids = filtered_client_ids

        # Handle for LASA's origin code if flat_local_updates is empty
        if len(flat_local_updates) == 0:
            print("  [LASA WARNING] All updates were corrupted (NaN/Inf) and filtered out. Skipping aggregation.")
            return global_model

        flat_all_grads = torch.stack(flat_local_updates, dim=0)
        grad_norm = torch.norm(flat_all_grads, dim=1).reshape((-1, 1))
        norm_clip = grad_norm.median(dim=0)[0].item()
        grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
        grads_clip = (flat_all_grads / grad_norm) * grad_norm_clipped

        del grad_norm, norm_clip, grad_norm_clipped

        clipped_local_updates = []

        for i in range(len(local_updates)):
            net = self._vector_to_net_dict(grads_clip[i], copy.deepcopy(local_updates[i]))
            clipped_local_updates.append(net)

        # Pre-aggregation sparsification
        for i in range(len(local_updates)):
            global_mask = generate_init_mask(local_updates[i])
            global_mask = update_mask(local_updates[i], global_mask, self.sparsity)
            local_updates[i] = apply_mask(local_updates[i], global_mask)

        key_mean_weight = {}
        for key in local_updates[0].keys():
            if 'num_batches_tracked' in key:
                continue
            key_flat_para = []
            all_set = set([i for i in range(len(local_updates))])
            for param in local_updates:
                flat_param = param[key].flatten()
                # print(flat_param.numel())
                key_flat_para.append(flat_param)
            grads = torch.stack(key_flat_para, dim=0)

            # Norm check
            grad_l2norm = torch.norm(grads.float(), dim=1).cpu().numpy()
            norm_med = np.median(grad_l2norm)
            norm_std = np.std(grad_l2norm)

            # Calculate MZ-score for Norm
            for i in range(len(grad_l2norm)):
                grad_l2norm[i] = np.abs((grad_l2norm[i] - norm_med) / (norm_std + 1e-9)) # Added 1e-9 to prevent division by zero
            
            benign_idx1 = all_set.copy()
            benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm < self.lambda_n)]))


            ##################
            # Sign check
            layer_sign = []
            for i in range(len(local_updates)):
                val = 0.5 * (1 + torch.sum(torch.sign(local_updates[i][key])) / (torch.sum(torch.abs(torch.sign(local_updates[i][key]))) + 1e-9) * (1 - self.sparsity)).item()
                layer_sign.append(val)

            benign_idx2 = all_set.copy()
            if len(layer_sign) > 0:
                median = np.median(layer_sign)
                std = np.std(layer_sign)

                # Calculate MZ-score
                for i in range(len(layer_sign)):
                    layer_sign[i] = np.abs((layer_sign[i] - median) / (std + 1e-9))
                benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(torch.tensor(layer_sign).cpu().numpy() < self.lambda_s)]))

            # Handle the case when both checks are too strict and filter out all clients
            benign_idx = list(benign_idx2.intersection(benign_idx1))
            if len(benign_idx) == 0:
                benign_idx = list(all_set)

            benign_idx = sorted([int(i) for i in benign_idx])

            # Save selected client indices for this layer
            self.selected_indices_by_layer[key] = benign_idx

            # Layer-wise adaptive aggregation
            key_mean_weight[key] = torch.mean(
                torch.stack([clipped_local_updates[i][key] for i in benign_idx], dim=0),
                dim=0
            )
        # Reset selected indices for this round based on the union of selected indices across all layers
        self.selected_indices = []
        for layer_key, selected_indices in self.selected_indices_by_layer.items():
            for idx in selected_indices:
                self.selected_indices.append(int(idx))
                    
        # Small adjustment from LASA's original code in update model
        new_global_state = copy.deepcopy(global_model.state_dict())
        for key in key_mean_weight.keys():
            if 'num_batches_tracked' in key:
                continue
            device = new_global_state[key].device
            new_global_state[key] += key_mean_weight[key].to(device)
        global_model.load_state_dict(new_global_state)
        
        return global_model