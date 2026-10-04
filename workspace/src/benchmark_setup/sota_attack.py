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

from helpers import log_and_print

class SOTAFLAttacks:
    def __init__(self, device="cpu", log_file_path=None):
        self.device = device
        self.log_file_path = log_file_path

    def apply_attack(self, attack_type: str, honest_deltas: list, **kwargs) -> dict:
        """
        Requires a list of all honest client deltas to compute the statistical bounds.
        Returns the crafted malicious DELTA to be shared by all colluding attackers.
        """
        if attack_type == "lie":
            return self._lie_attack(honest_deltas, kwargs.get("z_value", -0.5))
        elif attack_type == "min_max":
            return self._min_max_attack(honest_deltas, kwargs.get("gamma", 2.0))
        elif attack_type == "min_sum":
            return self._min_sum_attack(honest_deltas, kwargs.get("gamma", 2.0))
        elif attack_type == "byzmean":
            return self._byzmean_attack(honest_deltas)
        elif attack_type == "tailored_trmean":
            return self._tailored_trmean_attack(honest_deltas, kwargs.get("n_attackers", 1), kwargs.get("threshold", 10.0))
        else:
            raise ValueError(f"SOTA attack '{attack_type}' not recognized.")

    def _lie_attack(self, honest_deltas: list, z_value: float) -> dict:
        """
        "A Little Is Enough" (LIE) Attack.
        Calculates the mean (mu) and std (sigma) of honest updates.
        Crafts a malicious update exactly at: mu + z * sigma
        """
        log_and_print(f"Applying LIE attack with z-value: {z_value}", log_file_path=self.log_file_path)
        crafted_delta = {}
        param_names = honest_deltas[0].keys()

        for name in param_names:
            layer_updates = torch.stack([client[name] for client in honest_deltas])
            orig_dtype = layer_updates.dtype
            
            # Cast to float for statistical math
            float_updates = layer_updates.float()
            mu = torch.mean(float_updates, dim=0)
            sigma = torch.std(float_updates, dim=0)
            
            crafted_val = mu + (z_value * sigma)
            # Cast back to original type (to prevent integer mismatch errors)
            crafted_delta[name] = crafted_val.to(orig_dtype)

        return crafted_delta

    def _byzmean_attack(
        self,
        honest_deltas: list,
        num_total_clients: int,
        num_attackers: int,
        z: float = 0.5
    ) -> list:
        """
        Computes the Byzantine mean attack on honest client updates.

        Parameters:
        honest_deltas (list): List of update dictionaries from honest clients.
        num_total_clients (int): Total number of clients.
        num_attackers (int): Number of byzantine attackers (m1 + m2).
        z (float, optional): Z-value for determining the malicious update (default: 0.5).

        Attack details:
        - The attackers are split into two groups:
            - Group 1 (m1 attackers): Craft updates at mu - z*sigma to pull the mean down.
            - Group 2 (m2 attackers): Compute a coordinated update that, when combined with Group 1 and honest updates, 
            results in the overall mean being exactly at the malicious point.

        Returns:
        list: List of dictionaries, each containing the crafted malicious updates for a single attacker client.
        """
        print(f"Applying True ByzMean attack (z={z})")

        m1 = int(0.5 * num_attackers)
        m2 = num_attackers - m1

        mal_dict_1 = {}
        mal_dict_2 = {}

        param_names = honest_deltas[0].keys()

        for name in param_names:
            layer_updates = torch.stack([client[name] for client in honest_deltas])
            orig_dtype = layer_updates.dtype

            float_updates = layer_updates.float()

            # Honest statistics
            mu = torch.mean(float_updates, dim=0)
            sigma = torch.std(float_updates, dim=0)
            sigma = torch.nan_to_num(sigma, nan=0.0)

            # Target malicious update
            mal_update = mu - z * sigma
            mal_dict_1[name] = mal_update.to(orig_dtype)

            # Compute coordinated attackers (Group 2)
            if m2 > 0:
                sum_honest = torch.sum(float_updates, dim=0)
                # Solve:
                # (sum_honest + m1*mal_update + m2*x) / N = mal_update
                byz_grad2 = (
                    num_total_clients * mal_update
                    - sum_honest
                    - m1 * mal_update
                ) / m2

                mal_dict_2[name] = byz_grad2.to(orig_dtype)

        # Assign crafted update
        crafted_poisons = []

        for i in range(num_attackers):
            if i < m1:
                crafted_poisons.append({
                    k: v.clone() for k, v in mal_dict_1.items()
                })
            else:
                crafted_poisons.append({
                    k: v.clone() for k, v in mal_dict_2.items()
                })

        return crafted_poisons
    
    def _flatten_deltas(self, deltas: list):
        """Flattens a list of dictionary updates into a 2D tensor matrix."""
        flattened = []
        for update in deltas:
            flat = torch.cat([param.flatten() for param in update.values()])
            flattened.append(flat)
        return torch.stack(flattened)

    def _unflatten_delta(self, flat_delta, ref_dict):
        """Reconstructs the dictionary format from a flattened tensor."""
        unflattened = {}
        idx = 0
        for name, param in ref_dict.items():
            numel = param.numel()
            unflattened[name] = flat_delta[idx:idx+numel].view_as(param)
            idx += numel
        return unflattened
    
    def _min_max_attack(self, honest_deltas: list, threshold=10.0) -> dict:
        """
        Applies the exact NDSS Min-Max attack using Binary Search.
        Given the list of honest client deltas, computes the statistical bounds.
        Returns the crafted malicious DELTA to be shared by all colluding attackers.
        """
        print("Applying exact NDSS Min-Max attack (Binary Search)")
        all_updates_flat = self._flatten_deltas(honest_deltas).float()
        
        model_re = torch.mean(all_updates_flat, dim=0)
        deviation = torch.std(all_updates_flat, dim=0)
        
        # Calculate max distance between any two honest updates
        distances = torch.cdist(all_updates_flat, all_updates_flat, p=2) ** 2
        max_distance = torch.max(distances)
        
        lamda = torch.tensor(threshold, device=all_updates_flat.device)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0.0
        
        # Binary search for the maximum lambda
        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = model_re - lamda * deviation
            distance = torch.norm((all_updates_flat - mal_update), dim=1) ** 2
            max_d = torch.max(distance)
            
            if max_d <= max_distance:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2
            lamda_fail = lamda_fail / 2

        final_mal_update = model_re - lamda_succ * deviation
        return self._unflatten_delta(final_mal_update, honest_deltas[0])

    def _min_sum_attack(self, honest_deltas: list, threshold=10.0) -> dict:
        """
        Applies the exact NDSS Min-Sum attack using Binary Search.
        Given the list of honest client deltas, computes the statistical bounds.
        Returns the crafted malicious DELTA to be shared by all colluding attackers.
        """
        print("Applying exact NDSS Min-Sum attack (Binary Search)")
        all_updates_flat = self._flatten_deltas(honest_deltas).float()
        
        model_re = torch.mean(all_updates_flat, dim=0)
        deviation = torch.std(all_updates_flat, dim=0)
        
        # Calculate max sum of squared distances among honest updates
        distances = torch.cdist(all_updates_flat, all_updates_flat, p=2) ** 2
        scores = torch.sum(distances, dim=1)
        min_score = torch.max(scores) # Replicating paper's exact logic
        
        lamda = torch.tensor(threshold, device=all_updates_flat.device)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0.0
        
        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = model_re - lamda * deviation
            distance = torch.norm((all_updates_flat - mal_update), dim=1) ** 2
            score = torch.sum(distance)
            
            if score <= min_score:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2
            lamda_fail = lamda_fail / 2

        final_mal_update = model_re - lamda_succ * deviation
        return self._unflatten_delta(final_mal_update, honest_deltas[0])

    def _tailored_trmean_attack(self, honest_deltas: list, n_attackers: int, threshold=10.0) -> dict:
        """
        Applies the AGR-Tailored Trimmed Mean attack using Binary Search.
        Given the list of honest client deltas, computes the statistical bounds.
        Returns the crafted malicious DELTA to be shared by all colluding attackers.
        
        Parameters:
        honest_deltas (list): List of update dictionaries from honest clients.
        n_attackers (int): Number of byzantine attackers (m1 + m2).
        threshold (float, optional): Threshold value for the binary search. Defaults to 10.0.
        """
        
        print("Applying AGR-Tailored Trimmed Mean attack")
        all_updates_flat = self._flatten_deltas(honest_deltas).float()
        
        model_re = torch.mean(all_updates_flat, dim=0)
        deviation = torch.std(all_updates_flat, dim=0)
        
        lamda = torch.tensor(threshold, device=all_updates_flat.device)
        threshold_diff = 1e-5
        prev_loss = -1.0
        lamda_fail = lamda
        lamda_succ = 0.0
        
        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = model_re - lamda * deviation
            
            # Simulate the aggregation pool
            mal_updates_stack = torch.stack([mal_update] * n_attackers)
            combined_pool = torch.cat((mal_updates_stack, all_updates_flat), dim=0)
            
            # Simulate Trimmed Mean
            sorted_updates, _ = torch.sort(combined_pool, dim=0)
            agg_grads = torch.mean(sorted_updates[n_attackers:-n_attackers], dim=0)
            
            loss = torch.norm(agg_grads - model_re)
            
            if prev_loss < loss:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2
            
            lamda_fail = lamda_fail / 2
            prev_loss = loss

        final_mal_update = model_re - lamda_succ * deviation
        return self._unflatten_delta(final_mal_update, honest_deltas[0])
