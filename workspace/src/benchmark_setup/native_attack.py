"""
This file is derived from Jiahao Xu @ UNR's implementation.
Minor modifications have been made for this project.

Original source:
https://github.com/JianXu95/SignGuard/tree/main/aggregators (From Jiahao Xu @ UNR's cite)
https://github.com/JiiahaoXU/LASA/blob/master/algorithms/attack (By Jiahao Xu @ UNR)

Thanks for their contribution!

Modified by ppdddd et al.
"""

import torch
import torch.nn as nn

from helpers import log_and_print

class NaiveFLAttacks:
    def __init__(self, device="cpu", log_file_path=None):
        self.device = device
        self.log_file_path = log_file_path

    def apply_attack(self, attack_type: str, global_model: nn.Module, local_model: nn.Module, **kwargs) -> dict:
        """
        Returns the calculated malicious DELTA (not the full model) for easy aggregation.
        """
        if attack_type == "noise":
            return self._noise_attack(global_model, local_model, kwargs.get("noise_scale", 0.5))
        elif attack_type == "sign_flip":
            return self._sign_flip_attack(global_model, local_model, kwargs.get("flip_factor", -10.0))
        elif attack_type == "random":
            return self._random_attack(global_model, kwargs.get("random_factor", 0.5))
        else:
            raise ValueError(f"Naive attack '{attack_type}' not recognized.")

    def _sign_flip_attack(self, global_model: nn.Module, local_model: nn.Module, flip_factor: float) -> dict:
        log_and_print(f"Applying sign flipping attack with flip factor: {flip_factor}", log_file_path=self.log_file_path)
        malicious_delta = {}
        for name, param in global_model.state_dict().items():
            local_w = local_model.state_dict()[name].to(self.device)
            global_w = param.to(self.device)
            true_delta = local_w - global_w
            malicious_delta[name] = (true_delta * flip_factor).to(param.dtype)
            
        return malicious_delta

    def _noise_attack(self, global_model: nn.Module, local_model: nn.Module, scale: float) -> dict:
        log_and_print(f"Applying noise attack with scale: {scale}", log_file_path=self.log_file_path)
        malicious_delta = {}
        for name, param in global_model.state_dict().items():
            if 'num_batches_tracked' in name:
                malicious_delta[name] = torch.zeros_like(param)
                continue
                
            local_w = local_model.state_dict()[name].to(self.device)
            global_w = param.to(self.device)
            true_delta = local_w - global_w

            noisy_delta = true_delta + (torch.randn_like(true_delta.float()) * scale)
            malicious_delta[name] = noisy_delta.to(param.dtype)
            
        return malicious_delta

    def _random_attack(self, global_model: nn.Module, random_factor: float) -> dict:
        log_and_print("Applying random noise attack", log_file_path=self.log_file_path)
        malicious_delta = {}
        for name, param in global_model.state_dict().items():
            malicious_delta[name] = (random_factor * torch.randn_like(param.float())).to(param.dtype)
        return malicious_delta