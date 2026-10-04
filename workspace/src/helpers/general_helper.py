import warnings
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch
import copy

from .train_N_evaluate_ultils import train_client_locally

def log_and_print(*msgs, log_file_path=None):
    """
    Prints and logs the given messages to the specified log file.

    Args:
        *msgs: The messages to print and log.
        log_file: The path to the log file. Skip log if log file path is none

    Warnings raise:
        warnings: If log_file is None.
    """
    text = " ".join(str(m) for m in msgs)
    print(text)
    
    if log_file_path is None:
        warnings.warn("[WARN] [log_and_print] Log file is None skip saving log to file for message: " + text)
        return
    
    log_file = Path(log_file_path)
    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def set_seed(seed: int):
    """Locks all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Force deterministic operations (slightly slower, but guarantees exact reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_malicious_attacks(
    attack_category: str,
    global_model: nn.Module,
    client_datasets: dict,
    honest_clients: list,
    malicious_clients: list,
    client_updates: dict,
    n_clients: int,
    device: str,
    current_round: int,
    log_file_path: str = None
) -> dict:
    """
    Orchestrates the application of both Naive and SOTA attacks.
    Modifies and returns the client_updates dictionary.
    """
    actual_malicious_count = len(malicious_clients)

    if attack_category.lower() in ["cba", "dba", "none", "no_attack"]:
        return client_updates
    
    if not malicious_clients or attack_category == "skip":
        log_and_print("No attacks applied this round (attack_category='skip' or 0 attackers).", log_file_path=log_file_path)
        return client_updates

    if attack_category.lower() == "spa":

        valid_malicious = [cid for cid in malicious_clients if cid in client_updates]

        if len(valid_malicious) > 0:
            single_attacker_id = valid_malicious[0]
            
            total_data = sum([len(client_datasets[cid]) for cid in (honest_clients + malicious_clients)])
            ds = len(client_datasets[single_attacker_id])

            malicious_update = client_updates[single_attacker_id]
            
            scale_factor = total_data / ds
            delta = 5.0 # FedDLAD hardcode delta = 5 (base on FedDLAD's code file federated.py line 89-90)
            
            l2_norm_sq = 0.0
            for k, v in malicious_update.items():
                l2_norm_sq += torch.sum((v * scale_factor) ** 2).item()
            l2_norm = (l2_norm_sq ** 0.5)
            
            clip_ratio = delta / l2_norm if l2_norm > delta else 1.0
            final_multiplier = scale_factor * clip_ratio
            
            log_and_print(f"[SPA ATTACK] Attacker: {single_attacker_id} | Scale: {scale_factor:.2f} | Orig Norm: {l2_norm:.2f} | Final Multiplier: {final_multiplier:.2f}", log_file_path=log_file_path)

            for cid in valid_malicious:
                if cid == single_attacker_id:
                    crafted_update = {}
                    for k, v in malicious_update.items():
                        crafted_update[k] = v * final_multiplier
                    client_updates[cid] = crafted_update
                else:
                    zero_update = {k: torch.zeros_like(v) for k, v in malicious_update.items()}
                    client_updates[cid] = zero_update
                    
        return client_updates

    from benchmark_setup import NaiveFLAttacks, SOTAFLAttacks
    # --- NAIVE ATTACKS ---
    if attack_category in ["sign_flip", "noise", "random", "naive"]:
        naive_attacker = NaiveFLAttacks(device=device, log_file_path=log_file_path)
        specific_naive_attack = "sign_flip" if attack_category == "naive" else attack_category
        
        for cid in malicious_clients:
            local_model = copy.deepcopy(global_model)
            # Train locally to get the base delta, then corrupt it
            local_model = train_client_locally(local_model, client_datasets[cid], epochs=1, device=device, current_round=current_round)
            client_updates[cid] = naive_attacker.apply_attack(specific_naive_attack, global_model, local_model)
            log_and_print(f"Applied Naive {specific_naive_attack.upper()} attack for {cid}", log_file_path=log_file_path)

    # --- SOTA / ADVANCED ATTACKS ---
    elif attack_category in ["lie", "min_max", "min_sum", "byzmean", "tailored_trmean"]:
        sota_attacker = SOTAFLAttacks(device=device, log_file_path=log_file_path)
        honest_deltas_list = [client_updates[c] for c in honest_clients]
        
        if attack_category == "byzmean":
            crafted_poisons_list = sota_attacker._byzmean_attack(
                honest_deltas=honest_deltas_list, 
                num_total_clients=n_clients, 
                num_attackers=actual_malicious_count
            )
        else:
            single_poison = sota_attacker.apply_attack(
                attack_category, 
                honest_deltas_list,
                n_attackers=actual_malicious_count
            )
            crafted_poisons_list = [single_poison] * actual_malicious_count
        
        # Distribute the poisons
        for i, cid in enumerate(malicious_clients):
            client_updates[cid] = crafted_poisons_list[i]
            
        log_and_print(f"Applied SOTA {attack_category.upper()} attack for clients: {malicious_clients}", log_file_path=log_file_path)

    # If the attack category is unrecognized, log a warning but do not apply any attacks
    else:
        log_and_print(f"Attack category '{attack_category}' not recognized. No attacks applied.", log_file_path=log_file_path)

    return client_updates

def to_float_or_nan(x):
    if x is None:
        return np.nan
    if hasattr(x, "detach"):
        x = x.detach().cpu().item()
    elif hasattr(x, "item"):
        x = x.item()
    return float(x)

def safe_relative_change(value, baseline, eps=1e-12):
    value = float(value)
    baseline = float(baseline)

    if abs(baseline) < eps:
        return 0.0 if abs(value) < eps else np.nan

    return (value - baseline) / baseline


def compute_ops(acc1, asr1, acc2, asr2, is_fedavg=False, eps=1e-12):
    if is_fedavg:
        return 0.0

    acc1 = float(acc1)
    acc2 = float(acc2)
    asr1 = float(asr1)
    asr2 = float(asr2)

    acc_term = safe_relative_change(acc1, acc2, eps=eps)

    if abs(asr2) < eps:
        asr_term = 0.0 if abs(asr1) < eps else np.nan
    else:
        asr_term = (asr2 - asr1) / asr2

    if np.isnan(acc_term) or np.isnan(asr_term):
        return np.nan

    return acc_term + asr_term
    
def make_baseline_key(
    dataset_name,
    alpha,
    attack_category,
    update_method,
    percent_attackers,
    num_rounds,
    n_clients,
    clients_per_round,
    seed,
    poison_ratio,
):
    return (
        dataset_name,
        float(alpha),
        attack_category,
        update_method,
        float(percent_attackers),
        int(num_rounds),
        int(n_clients),
        int(clients_per_round),
        int(seed),
        float(poison_ratio),
    )

def calculate_tpr_at_fixed_fpr(y_true, y_score, target_fpr=0.05):
    """
    Calculates the TPR matching an exact FPR target using interpolation.
    y_true: 1 for malicious, 0 for honest
    y_score: Anomaly score (e.g., 1.0 - weight)
    """
    if len(set(y_true)) <= 1:
        return np.nan
        
    from sklearn.metrics import roc_curve
    fprs, tprs, thresholds = roc_curve(y_true, y_score)
    
    # Interpolate the TPR value at exactly the target FPR point
    tpr_at_target = float(np.interp(target_fpr, fprs, tprs))
    return tpr_at_target * 100.0 # Convert to percentage