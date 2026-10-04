import numpy as np
from typing import Dict, Literal
import torch.nn as nn

from tsfl.tsfl_helpers.tsfl_tensor_helper import flatten_client_updates_to_1d_numpy

class UpdateDirectionEvaluator:
    def __init__(
            self, 
            aggregation_method: Literal["mean", "median"] = "median", 
            eps: float = 1e-8,
            lower_scale_mad_for_cosine: float = 0.05
        ):
        self.aggregation_method = aggregation_method
        self.eps = eps
        self.lower_scale_mad_for_cosine = lower_scale_mad_for_cosine

    def validate_directions(self, client_updates: Dict[str, dict],  global_model: nn.Module, log_file_path: str = None) -> dict:
        # Flatten all client dictionaries into 1D arrays
        param_keys = []
        param_shapes = {}

        for k, v in global_model.named_parameters():
            param_keys.append(k)
            param_shapes[k] = v.shape
        
        flattened_dict = flatten_client_updates_to_1d_numpy(
            client_updates_need_to_flatten=client_updates, 
            param_keys=param_keys,
            param_shapes=param_shapes,
            log_file_path=log_file_path
        )
        ids = list(flattened_dict.keys())

        # Stack all update vectors into a matrix of shape (n, d) for n clients and d dimensions
        update_vectors_list = []
        for client_id in ids:
            update_vectors_list.append(flattened_dict[client_id])
        stacked_update_matrix = np.vstack(update_vectors_list)

        # Compute the global update direction
        if self.aggregation_method == "median":
            # Element-wise median is highly robust to attackers
            global_reference = np.median(stacked_update_matrix, axis=0)
        else:
            global_reference = np.mean(stacked_update_matrix, axis=0)
        
        # Calculate norm
        ref_norm = np.linalg.norm(global_reference)
        
        # Fallback to mean if median collapsed
        if ref_norm < self.eps:
            global_reference = np.mean(stacked_update_matrix, axis=0)
            ref_norm = np.linalg.norm(global_reference)
        global_reference = global_reference / (ref_norm + self.eps)

        # Compute sign disagreement ratio between each client's update and the global reference
        global_sign = np.sign(global_reference)
        client_signs = np.sign(stacked_update_matrix)

        sign_disagreements = np.mean(client_signs != global_sign, axis=1)
        med_sign = float(np.median(sign_disagreements))
        mad_sign = float(np.median(np.abs(sign_disagreements - med_sign)))
        safe_mad_sign = max(mad_sign, 0.01)

        # Compute the cosine similarity of each client's update against the global reference
        client_norms = np.linalg.norm(stacked_update_matrix, axis=1, keepdims=True)
        normalized_matrix = stacked_update_matrix / (client_norms + self.eps)
        cos_similarities_array = np.dot(normalized_matrix, global_reference)
            
        # Compute statistics
        med_cos = float(np.median(cos_similarities_array)) # The median cosine similarity
        mad_cos = float(np.median(np.abs(cos_similarities_array - med_cos))) # The MAD
        safe_scale = max(mad_cos, self.lower_scale_mad_for_cosine)

        # Compute relative rank for each client based on cosine similarity
        rank = np.argsort(np.argsort(cos_similarities_array))
        
        
        # Build the final dictionary in one clean pass
        client_scores = {}
        for i, client_id in enumerate(ids):
            cos_sim = float(cos_similarities_array[i])
            z_score = (cos_sim - med_cos) / safe_scale
            relative_rank = rank[i] / (len(ids) - 1 + 1e-8)

            sign_disagreement = float(sign_disagreements[i])
            sign_z_score = (sign_disagreement - med_sign) / safe_mad_sign
            
            client_scores[client_id] = {
                "global_deviation_z_score": float(z_score),
                "direction_anomaly_score": float(np.tanh(abs(z_score))),
                "cosine_rank": float(relative_rank),
                
                "sign_disagreement_ratio": sign_disagreement,
                "sign_anomaly_score": float(np.tanh(abs(sign_z_score)))
            }

        return {
            "keep_ids": ids,
            "client_scores": client_scores,

            "batch_median_cosine": med_cos,
            "batch_mad_cosine": mad_cos
        }