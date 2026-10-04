import numpy as np
from typing import Dict
import torch.nn as nn

from tsfl.tsfl_helpers.tsfl_tensor_helper import flatten_client_updates_to_1d_numpy

class UpdateScaleValidator:
    def __init__(self, tolerance_multiplier: float = 3.0, fallback_threshold: float = 1e-5):
        self.tolerance_multiplier = tolerance_multiplier
        self.fallback_threshold = fallback_threshold

    def validate_scales(self, client_updates: Dict[str, dict],  global_model: nn.Module, log_file_path: str = None) -> dict:
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
        
        # Compute the L2 Norm (magnitude) for each client
        client_norms = {}
        norm_values = []
        for client_id in ids:
            vector = flattened_dict[client_id]
            magnitude = float(np.linalg.norm(vector)) # L2 calculation
            client_norms[client_id] = magnitude
            norm_values.append(magnitude)

        norm_array = np.array(norm_values)

        # Calculate the threshold using Median Absolute Deviation (MAD)
        # MAD = median(abs(x - median(x)))
        median_norm = np.median(norm_array)
        mad = np.median(np.abs(norm_array - median_norm))
        
        # If MAD is 0 (all clients have the exact same norm), use the fallback to avoid dropping everyone
        safe_mad = max(mad, self.fallback_threshold)

        client_scores = {}
        for client_id, norm_val in client_norms.items():
            z_score = (norm_val - median_norm) / (safe_mad + 1e-8)
            client_scores[client_id] = {
                "scale_zscore": float(z_score),
                "scale_anomaly_score": float(np.tanh(abs(z_score)))
            }

        return {
            "keep_ids": ids,
            "client_scores": client_scores,

            "median_norm": float(median_norm),
            "mad": float(safe_mad),
        }