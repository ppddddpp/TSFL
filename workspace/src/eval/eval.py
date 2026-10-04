import os
import torch

torch._dynamo.config.suppress_errors = True
from torch.utils.data import DataLoader, Subset
import copy
from pathlib import Path
import numpy as np
import os
import random
import pandas as pd
import time
import json
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score

from tsfl import TsflMain, TsflFeatureValidator
from benchmark_setup.fl_data_benchmark import DirichletDataPartitioner
from helpers import log_and_print, set_seed, to_float_or_nan, compute_ops, make_baseline_key
from helpers import compute_client_update, evaluate_model, train_client_locally
from helpers import apply_malicious_attacks
from helpers import calculate_tpr_at_fixed_fpr
from helpers import TSFLReport
from benchmark_setup.data_and_model_setup import DataModelSetup

from other_defenses import OtherDefensesSetup

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR_GRAPH = OUTPUT_DIR / "output_graphs"
OUTPUT_DIR_CSV = OUTPUT_DIR / "output_csv"
OUTPUT_DIR_LOG = OUTPUT_DIR / "output_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_GRAPH, exist_ok=True)
os.makedirs(OUTPUT_DIR_CSV, exist_ok=True)

# Global variables to store baselines and OPS results for analysis
FEDAVG_BASELINES = {}
OPS_RESULTS = {}


def eval_full_pipeline(
        dataset_name: str = "cifar10",
        defense_name: str = "all",
        alpha: float = 0.1, 
        attack_category: str = "byzmean", 
        update_method: str = "fedavg",
        percent_attackers: float = 0.25, 
        num_rounds: int = 100,
        n_clients: int = 100,
        clients_per_round: int = 25,
        seed: int = 2709,
        batch_size: int = 64,
        poison_ratio: float = 0.5,
    ):
    if defense_name.lower() != "all":
        defenses_to_test = [defense_name.lower().strip()]
    else:
        defenses_to_test = OtherDefensesSetup().get_built_in_defenses()
        fedavg_defenses = [d for d in defenses_to_test if d.lower().strip() == "fedavg"]
        other_defenses = [d for d in defenses_to_test if d.lower().strip() != "fedavg"]
        defenses_to_test = fedavg_defenses + other_defenses
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"AUTO BENCHMARK PIPELINE")
    print(f"Defenses queued: {defenses_to_test}")
    print(f"{'='*60}\n")

    for defense in defenses_to_test:
        print(f"\n{'*'*50}")
        print(f" STARTING RUN FOR: {defense.upper()}")
        print(f"{'*'*50}\n")
        
        # Run the isolated pipeline
        final_acc, total_runtime = eval_single_pipeline(
            dataset_name=dataset_name, 
            alpha=alpha, 
            attack_category=attack_category,
            defense_name=defense, 
            update_method=update_method, 
            percent_attackers=percent_attackers, 
            num_rounds=num_rounds, 
            n_clients=n_clients, 
            clients_per_round=clients_per_round, 
            seed=seed, 
            batch_size=batch_size,
            poison_ratio=poison_ratio,
        )
        results[defense] = {
            "acc": final_acc,
            "runtime": total_runtime
        }

    # Print Final Leaderboard
    baseline_key = make_baseline_key(
        dataset_name=dataset_name,
        alpha=alpha,
        attack_category=attack_category,
        update_method=update_method,
        percent_attackers=percent_attackers,
        num_rounds=num_rounds,
        n_clients=n_clients,
        clients_per_round=clients_per_round,
        seed=seed,
        poison_ratio=poison_ratio,
    )
    
    print("\n" + "="*80)
    print("AUTO BENCHMARK FINAL LEADERBOARD")
    print("="*80)
    
    leaderboard_data = []
    for d, metrics in sorted(results.items(), key=lambda item: item[1]["acc"], reverse=True):
        defense_key = d.lower().strip()
        acc = metrics["acc"]
        runtime = metrics["runtime"]
        
        ops_info = OPS_RESULTS.get((baseline_key, defense_key), {})
        mean_ops = ops_info.get("mean_round_ops", np.nan)
        last_ops = ops_info.get("last_round_ops", np.nan)
        mean_asr = ops_info.get("mean_round_asr", np.nan)
        last_asr = ops_info.get("last_round_asr", np.nan)

        print(
            f"{d.upper():<15}: "
            f"BestAcc={acc:.2f}% | "
            f"MeanASR={mean_asr:.2f} | "
            f"LastASR={last_asr:.2f} | "
            f"MeanOPS={mean_ops:.2f} | "
            f"LastOPS={last_ops:.2f} | "
            f"Runtime={runtime/60:.1f} mins"
        )
        
        leaderboard_data.append({
            "Defense": d.upper(),
            "Best_Accuracy": acc,
            "Mean_ASR": mean_asr,
            "Last_ASR": last_asr,
            "Mean_OPS": mean_ops,
            "Last_OPS": last_ops,
            "Runtime_Seconds": runtime,
            "Runtime_Minutes": runtime / 60
        })
    print("="*80)
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    csv_leaderboard_path = OUTPUT_DIR_CSV / f"leaderboard_{dataset_name}_alpha_{alpha}_{attack_category}_ratio_{percent_attackers}_seed_{seed}_poison_{poison_ratio}.csv"
    leaderboard_df.to_csv(csv_leaderboard_path, index=False)
    print(f"[INFO] Leaderboard saved to: {csv_leaderboard_path}")
    
    return results

def eval_single_pipeline(
    dataset_name: str = "cifar10",
    alpha: float = 0.1, 
    attack_category: str = "byzmean", 
    defense_name: str = "tsfl",
    update_method: str = "fedavg",
    percent_attackers: float = 0.25, 
    num_rounds: int = 100,
    n_clients: int = 100,
    clients_per_round: int = 25,
    seed: int = 2709,
    batch_size: int = 64,
    poison_ratio: float = 0.5,
    ):
    pipeline_start_time = time.time()

    dynamic_output_dir = BASE_DIR / "output" / f"case_data_{dataset_name.lower()}"
    dynamic_csv_dir = dynamic_output_dir / "output_csv"
    dynamic_log_dir = dynamic_output_dir / "output_logs"
    dynamic_graph_dir = dynamic_output_dir / "output_graphs"

    os.makedirs(dynamic_csv_dir, exist_ok=True)
    os.makedirs(dynamic_log_dir, exist_ok=True)
    os.makedirs(dynamic_graph_dir, exist_ok=True)

    # Initialize global variables for storing baselines and OPS results
    global FEDAVG_BASELINES, OPS_RESULTS
    
    log_file_path = dynamic_log_dir / f"{defense_name}_alpha_{alpha}_attack_{attack_category}_ratio_{percent_attackers}_seed_{seed}_poison_{poison_ratio}.log"
    log_and_print(f"\n{'=' * 150}", log_file_path=log_file_path)

    msg=f"Running FL Simulation | Alpha: {alpha} | Attack: {attack_category.upper()} | Defense: {defense_name.upper()} " \
        + f"Update Method: {update_method.upper()}| Attack Ratio: {percent_attackers*100:.1f}% " \
        + f"| Seed: {seed} | Clients: {n_clients} | Rounds: {num_rounds}"
    
    log_and_print(msg, log_file_path=log_file_path)
    log_and_print(f"=" * 150, log_file_path=log_file_path)
    
    log_and_print(f"Setting random seed to {seed} for reproducibility.", log_file_path=log_file_path)
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    log_and_print(f"Running FL Simulation on device: {device}", log_file_path=log_file_path)

    data_dir = BASE_DIR / "data"
    data_model_setup = DataModelSetup(data_dir)
    train_dataset, test_dataset = data_model_setup.get_dataset(dataset_name=dataset_name)

    if hasattr(test_dataset, 'classes'):
        num_classes = len(test_dataset.classes)
    else:
        class_mapping = {"cifar10": 10, "cifar100": 100}
        num_classes = class_mapping.get(dataset_name.lower(), 10)

    global_model = data_model_setup.get_model(dataset_name=dataset_name, device=device)
    global_model = global_model.to(device)
    shared_local_model = copy.deepcopy(global_model)

    log_and_print("Compiling Global Model...", log_file_path=log_file_path)
    compiled_client_model = torch.compile(shared_local_model)

    log_and_print("Preparing Balanced Server Proxy Set...", log_file_path=log_file_path)
    targets = torch.as_tensor(train_dataset.targets)
    proxy_indices = []
    samples_per_class = 10
    
    for class_id in torch.unique(targets).tolist():
        class_indices = torch.where(targets == class_id)[0]
        proxy_indices.extend(class_indices[:samples_per_class].tolist())
        
    proxy_dataset = Subset(train_dataset, proxy_indices)

    if defense_name.lower().strip() == "fltrust":
        server_proxy_loader = DataLoader(
            proxy_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
    elif defense_name.lower().strip() == "tsfl":
        server_proxy_loader = DataLoader(
            proxy_dataset,
            batch_size=len(proxy_indices),
            shuffle=False,
        )
    else:
        server_proxy_loader = None

    client_pool_indices = [i for i in range(len(train_dataset)) if i not in proxy_indices]
    
    class ClientPoolDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, indices):
            self.original_dataset = original_dataset
            self.indices = indices
            self.targets = [original_dataset.targets[i] for i in indices]
            if hasattr(original_dataset, 'classes'):
                self.classes = original_dataset.classes

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.original_dataset[self.indices[idx]]

    client_train_dataset = ClientPoolDataset(train_dataset, client_pool_indices)

    partitioner = DirichletDataPartitioner(dataset=client_train_dataset, n_clients=n_clients, alpha=alpha, seed=seed)
    client_datasets = partitioner.generate_client_datasets(log_file_path=log_file_path)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    log_and_print("Pushing Test Set to VRAM...", log_file_path=log_file_path)
    test_images_list, test_labels_list = [], []
    for img, lbl in test_loader:
        test_images_list.append(img.to(device, non_blocking=True))
        test_labels_list.append(lbl.to(device, non_blocking=True))
    
    gpu_test_images = torch.cat(test_images_list)
    gpu_test_labels = torch.cat(test_labels_list)

    # Mocking data for the skeleton script
    client_ids = [f"client_{i}" for i in range(n_clients)]
    actual_malicious_count = int(round(percent_attackers * n_clients))
    estimated_attackers_per_round = int(round(percent_attackers * clients_per_round))

    if defense_name == "tsfl":
        validator = TsflFeatureValidator(dataset_loader=server_proxy_loader)
        active_defense = TsflMain(tsfl_feature_validator=validator)
    else:
        defense_factory = OtherDefensesSetup()
        active_defense = defense_factory.define_defenses(
            defense_name=defense_name,
            dataset_name=dataset_name,
            clients_per_round=clients_per_round,
            server_proxy_loader=server_proxy_loader
        )
    
    rng = random.Random(seed) 
    shuffled_ids = client_ids.copy()
    rng.shuffle(shuffled_ids)
    
    malicious_clients = shuffled_ids[:actual_malicious_count]

    log_and_print(f"Attack ratio: {percent_attackers*100:.1f}% | Actual attackers (Total pool): {actual_malicious_count}", log_file_path=log_file_path)
    log_and_print(f"Malicious clients (randomized): {malicious_clients}", log_file_path=log_file_path)

    history = {
        "round": [], 
        "mta": [], 
        "precision": [],
        "recall": [],
        "f1": [],
        "honest_avg_weight": [], 
        "malicious_avg_weight": [],
        "tpr": [],
        "fpr": [],
        "tpr_5": [],
        "tpr_10": [],
        "auc": [],
        "ap": [],
        "attack_num": [],
        "attack_pass": [],
        "attack_ratio": [],
        "ops": [],
        "fedavg_acc_baseline": [],
        "fedavg_asr_baseline": [],
    }
    best_accuracy_so_far = 0.0
    best_round_stats = {"round": 0, "energies": {}, "soft_weights": {}, "weights": {}, "val_out": {}}
    validation_backup = {}
    
    log_and_print(f"\n=== Do Federated Learning for {num_rounds} rounds ===", log_file_path=log_file_path)
    client_weight_rows = []
    
    baseline_key = make_baseline_key(
        dataset_name=dataset_name,
        alpha=alpha,
        attack_category=attack_category,
        update_method=update_method,
        percent_attackers=percent_attackers,
        num_rounds=num_rounds,
        n_clients=n_clients,
        clients_per_round=clients_per_round,
        seed=seed,
        poison_ratio=poison_ratio,
    )

    defense_key = defense_name.lower().strip()

    baseline_filepath = dynamic_csv_dir / f"fedavg_baseline_{dataset_name}_alpha_{alpha}_attack_{attack_category}_ratio_{percent_attackers}_seed_{seed}_poison_{poison_ratio}.json"
    if defense_key != "fedavg":
        if baseline_filepath.exists():
            with open(baseline_filepath, "r") as f:
                FEDAVG_BASELINES[baseline_key] = json.load(f)
            log_and_print(f"[INFO] Successfully loaded FedAvg baseline from cache for OPS computation.", log_file_path=log_file_path)
        else:
            log_and_print(f"[WARNING] FedAvg baseline NOT FOUND at {baseline_filepath}. OPS will be NaN. You must run '--defense fedavg' first!", log_file_path=log_file_path)
    
    for round_num in range(1, num_rounds + 1):
        log_and_print(f"\n--- Round {round_num}/{num_rounds} ---", log_file_path=log_file_path)
        global_state = copy.deepcopy(global_model.state_dict())
        client_updates = {}

        selected_clients_this_round = random.sample(shuffled_ids, clients_per_round)

        honest_clients = [cid for cid in selected_clients_this_round if cid not in malicious_clients]
        malicious_clients_this_round = [cid for cid in selected_clients_this_round if cid in malicious_clients]

        # Local Client Training
        for cid in selected_clients_this_round:
            shared_local_model.load_state_dict(global_state)

            # HONEST CLIENTS: Train normally on clean data
            if cid in honest_clients:
                train_client_locally(
                    model=compiled_client_model, 
                    dataset=client_datasets[cid], 
                    current_round=round_num, 
                    epochs=3, 
                    device=device
                )
                client_updates[cid] = compute_client_update(global_model, shared_local_model)

            # TARGETED ATTACKERS (CBA, DBA, SPA): Train on Poisoned Data
            elif cid in malicious_clients_this_round and attack_category.lower() in ["cba", "dba", "spa"]:
                log_and_print(f"[{attack_category.upper()}] Client {cid} training on poisoned dataset.", log_file_path=log_file_path)

                numeric_agent_idx = int(cid.split('_')[1])
                
                from benchmark_setup import PoisonedClientDataset
                poisoned_dataset = PoisonedClientDataset(
                    original_subset=client_datasets[cid], 
                    dataset_name=dataset_name,
                    attack_type="cba" if attack_category.lower() == "spa" else attack_category, # SPA uses CBA trigger before clipping
                    agent_idx=numeric_agent_idx,
                    base_class=0,
                    target_class=5,
                    poison_ratio=poison_ratio
                )
                
                train_client_locally(
                    model=compiled_client_model,
                    dataset=poisoned_dataset,
                    current_round=round_num, 
                    epochs=3, 
                    device=device
                )
                client_updates[cid] = compute_client_update(global_model, shared_local_model)
            
            else:
                pass

        # Apply Untargeted Attacks (MALICIOUS CLIENTS ONLY)
        client_updates = apply_malicious_attacks(
            attack_category=attack_category,
            global_model=global_model,
            client_datasets=client_datasets,
            honest_clients=honest_clients,
            malicious_clients=malicious_clients_this_round, 
            client_updates=client_updates,
            n_clients=clients_per_round,
            device=device,
            current_round=round_num,
            log_file_path=log_file_path
        )

        # Analyze Client Updates
        log_and_print(f"Analyzing client updates via {defense_name} Defense Pipeline...", log_file_path=log_file_path)

        honest_updates, val_out, final_weights, raw_energies, soft_weights = {}, {}, {}, {}, {}

        # Defense and Anomaly Detection
        try:
            selected_data_sizes = {
                cid: len(client_datasets[cid])
                for cid in client_updates.keys()
            }

            if defense_name == "tsfl":
                honest_updates, val_out, final_weights, raw_energies, soft_weights = active_defense.analyze(
                    global_model=global_model, 
                    client_updates=client_updates,
                    log_file_path=log_file_path,
                    device=device,
                )
                
                # Secure aggregation
                global_model = active_defense.update_weights(
                    update_method=update_method, 
                    global_model=global_model, 
                    client_updates=honest_updates, 
                    soft_weights=final_weights
                )
            elif defense_key == "feddlad":
                selected_data_sizes = {
                    cid: len(client_datasets[cid])
                    for cid in client_updates.keys()
                }
                global_model = active_defense.run(
                    global_model=global_model, 
                    client_updates_dict=client_updates,
                    client_data_sizes=selected_data_sizes
                )
            elif defense_key == "fltrust":
                global_model = active_defense.run(
                    global_model=global_model, 
                    client_updates_dict=client_updates,
                    current_round=round_num,
                )
            else:
                global_model = active_defense.run(
                    global_model=global_model, 
                    client_updates_dict=client_updates
                )
                
        except RuntimeError as e:
            if "non-empty TensorList" in str(e) or "NaN" in str(e):
                msg = f"CRITICAL FAILURE: {defense_name.upper()} collapsed due to Numerical Instability (NaN generated). Halting simulation for this defense."
                log_and_print(msg, log_file_path=log_file_path)
                break
            else:
                raise e
        
        if defense_name == "tsfl":
            round_rows = []

            for cid in selected_clients_this_round:
                round_rows.append({
                    "Round": round_num,
                    "Client_id": cid,
                    "energies": to_float_or_nan(raw_energies.get(cid)),
                    "soft_weights": to_float_or_nan(soft_weights.get(cid)),
                    "weights": to_float_or_nan(final_weights.get(cid)),
                })

            round_rows = sorted(
                round_rows,
                key=lambda x: x["weights"] if not np.isnan(x["weights"]) else -np.inf,
                reverse=True
            )

            client_weight_rows.extend(round_rows)
            
        # Generic attack success-rate analysis for OPS
        actual_k = len(client_updates)

        avg_weight_threshold = (
            1.0 / actual_k
            if actual_k > 0
            else 0.0
        )
        num_attackers_this_round = len(malicious_clients_this_round)
        defense_key = defense_name.lower().strip()

        if num_attackers_this_round == 0:
            attacker_success_count = 0

        elif defense_key == "tsfl":
            attacker_success_count = sum(
                1
                for cid in malicious_clients_this_round
                if to_float_or_nan(final_weights.get(cid, 0.0)) > avg_weight_threshold
            )
        elif defense_key in ["fltrust", "foolsgold"]:
            attacker_success_count = sum(
                1 for cid in malicious_clients_this_round
                if active_defense.client_weights.get(cid, 0.0) > avg_weight_threshold
            )
        elif defense_key == "lasa":
            selected_indices_by_layer = getattr(active_defense, "selected_indices_by_layer", None)

            if selected_indices_by_layer is not None and len(selected_indices_by_layer) > 0:
                update_client_ids = getattr(
                    active_defense,
                    "filtered_client_ids",
                    list(client_updates.keys())
                )

                malicious_set = set(malicious_clients_this_round)
                num_layers = len(selected_indices_by_layer)

                attacker_success_count = 0

                for layer_key, selected_indices in selected_indices_by_layer.items():
                    selected_ids_this_layer = {
                        update_client_ids[int(i)]
                        for i in selected_indices
                        if 0 <= int(i) < len(update_client_ids)
                    }

                    attacker_success_count += len(
                        selected_ids_this_layer.intersection(malicious_set)
                    )
            
                num_attackers_this_round = num_attackers_this_round * num_layers
            else:
                attacker_success_count = num_attackers_this_round
                
        elif defense_key == "fedavg":
            # FedAvg has no defense, so all malicious clients are considered to pass.
            attacker_success_count = num_attackers_this_round

        else:
            # For filtering-based defenses such as Krum, Bulyan, DnC, SignGuard
            # use selected_indices if available.
            update_client_ids = list(client_updates.keys())
            selected_indices = getattr(active_defense, "selected_indices", None)

            if selected_indices is not None:
                selected_ids = {
                    update_client_ids[int(i)]
                    for i in selected_indices
                    if 0 <= int(i) < len(update_client_ids)
                }

                attacker_success_count = sum(
                    1 for cid in malicious_clients_this_round
                    if cid in selected_ids
                )
            else:
                # If the defense has no explicit filtering output,
                # assume attackers are not explicitly blocked.
                attacker_success_count = num_attackers_this_round

        attack_ratio_this_round = (
            attacker_success_count / num_attackers_this_round
            if num_attackers_this_round > 0
            else 0.0
        )

        history["attack_num"].append(num_attackers_this_round)
        history["attack_pass"].append(attacker_success_count)
        history["attack_ratio"].append(attack_ratio_this_round)

        log_and_print(
            f"Attack Success Rate | Avg weight threshold: {avg_weight_threshold:.2f} "
            f"| Attackers passed: {attacker_success_count}/{num_attackers_this_round} "
            f"| ASR: {attack_ratio_this_round:.2f}",
            log_file_path=log_file_path
        )
        # Global Evaluation
        eval_metrics = evaluate_model(global_model, gpu_test_images, gpu_test_labels, num_classes, device)
        current_accuracy = eval_metrics["Accuracy"]
        
        if defense_key == "fedavg":
            round_ops = 0.0
            fedavg_acc_round = current_accuracy
            fedavg_asr_round = attack_ratio_this_round

        else:
            fedavg_baseline = FEDAVG_BASELINES.get(baseline_key)

            if fedavg_baseline is None:
                round_ops = np.nan
                fedavg_acc_round = np.nan
                fedavg_asr_round = np.nan

                log_and_print(
                    "OPS Warning | FedAvg baseline is not available yet for this configuration.",
                    log_file_path=log_file_path
                )

            else:
                round_idx = round_num - 1

                if round_idx < len(fedavg_baseline["round_acc"]):
                    fedavg_acc_round = fedavg_baseline["round_acc"][round_idx]
                    fedavg_asr_round = fedavg_baseline["round_asr"][round_idx]

                    round_ops = compute_ops(
                        acc1=current_accuracy,
                        asr1=attack_ratio_this_round,
                        acc2=fedavg_acc_round,
                        asr2=fedavg_asr_round,
                        is_fedavg=False,
                    )
                else:
                    round_ops = np.nan
                    fedavg_acc_round = np.nan
                    fedavg_asr_round = np.nan
        
        history["round"].append(round_num)
        history["mta"].append(current_accuracy)
        history["precision"].append(eval_metrics["Precision"])
        history["recall"].append(eval_metrics["Recall"])
        history["f1"].append(eval_metrics["F1_Score"])
        history["ops"].append(round_ops)
        history["fedavg_acc_baseline"].append(fedavg_acc_round)
        history["fedavg_asr_baseline"].append(fedavg_asr_round)

        if defense_name == "tsfl":
            h_weights = [final_weights.get(c, 0.0) for c in honest_clients]
            m_weights = [final_weights.get(c, 0.0) for c in malicious_clients_this_round] 
            history["honest_avg_weight"].append(np.mean(h_weights) if h_weights else 0.0)
            history["malicious_avg_weight"].append(np.mean(m_weights) if m_weights else 0.0)

            # --- ADAPTIVE THRESHOLD ---
            all_weights = np.array([final_weights.get(cid, 0.0) for cid in selected_clients_this_round])
            
            # Check the spread of weights. 
            if np.std(all_weights) > 1e-4:
                weight_matrix = all_weights.reshape(-1, 1)

                # Fit 1-Cluster Model
                gmm_1 = GaussianMixture(n_components=1, random_state=seed).fit(weight_matrix)
                bic_1 = gmm_1.bic(weight_matrix)

                # Fit 2-Cluster Model
                gmm_2 = GaussianMixture(n_components=2, random_state=seed).fit(weight_matrix)
                bic_2 = gmm_2.bic(weight_matrix)

                # Information Theory Decision: 1 Cluster or 2?
                if bic_1 < bic_2:
                    # The math dictates this is a single, unified distribution.
                    neutralization_threshold = -1.0 
                else:
                    # The math dictates there are two distinct distributions.
                    centroids = gmm_2.means_.flatten()
                    neutralization_threshold = float(np.mean(centroids))
            else:
                # All weights are identical (Zero Variance). Bypass GMM.
                neutralization_threshold = -1.0

            # --- TPR / FPR CALCULATION ---
            if malicious_clients_this_round:
                detected_malicious = sum(1 for w in m_weights if w <= neutralization_threshold)
                tpr = (detected_malicious / len(malicious_clients_this_round)) * 100.0
            else:
                tpr = 100.0
                
            if honest_clients:
                falsely_detected_honest = sum(1 for w in h_weights if w <= neutralization_threshold)
                fpr = (falsely_detected_honest / len(honest_clients)) * 100.0
            else:
                fpr = 0.0
                
            history["tpr"].append(tpr)
            history["fpr"].append(fpr)

            # --- AUC / AP / TPR@FPR CALCULATION ---
            y_true = []
            y_score = []

            for cid in selected_clients_this_round:
                y_true.append(1 if cid in malicious_clients_this_round else 0)
                # Lower weight means higher suspicion (anomaly score)
                y_score.append(1.0 - final_weights.get(cid, 0.0))

            # ROC-AUC and TPR@FPR require at least one positive and one negative sample
            if len(set(y_true)) > 1:
                auc = roc_auc_score(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                tpr_5 = calculate_tpr_at_fixed_fpr(y_true, y_score, target_fpr=0.05)
                tpr_10 = calculate_tpr_at_fixed_fpr(y_true, y_score, target_fpr=0.10)
            else:
                auc = np.nan
                ap = np.nan
                tpr_5 = np.nan
                tpr_10 = np.nan

            history["auc"].append(auc)
            history["ap"].append(ap)
            history["tpr_5"].append(tpr_5)
            history["tpr_10"].append(tpr_10)

        else:
            history["honest_avg_weight"].append(0.0)
            history["malicious_avg_weight"].append(0.0)
            history["tpr"].append(0.0)
            history["fpr"].append(0.0)
            history["auc"].append(np.nan)
            history["ap"].append(np.nan)
            history["tpr_5"].append(np.nan)
            history["tpr_10"].append(np.nan)
        
        log_and_print(f"Round {round_num} Completed | Global accuracy: {current_accuracy:.2f}%", log_file_path=log_file_path)
        if defense_name == "tsfl":
            log_and_print(f"   => TSFL Sensor: AUC = {auc:.2f} | AP = {ap:.2f} | TPR@0.05 = {tpr_5:.2f} | TPR@0.10 = {tpr_10:.2f}", log_file_path=log_file_path)
            log_and_print(f"   => TSFL Sensor: TPR = {tpr:.1f}% | FPR = {fpr:.1f}%", log_file_path=log_file_path)
        log_and_print(
            f"Round OPS | ASR: {attack_ratio_this_round:.2f} "
            f"| FedAvg Acc: {fedavg_acc_round:.2f} "
            f"| FedAvg ASR: {fedavg_asr_round:.2f} "
            f"| OPS: {round_ops:.2f}",
            log_file_path=log_file_path
        )
        validation_backup[round_num] = copy.deepcopy(val_out)

        if current_accuracy > best_accuracy_so_far:
            best_accuracy_so_far = current_accuracy
            
            if defense_name == "tsfl":
                best_round_stats["energies"] = copy.deepcopy(raw_energies)
                best_round_stats["soft_weights"] = copy.deepcopy(soft_weights)
                best_round_stats["weights"] = copy.deepcopy(final_weights)
                best_round_stats["val_out"] = copy.deepcopy(val_out)

        client_updates.clear()
        honest_updates.clear()
        val_out.clear()
        final_weights.clear()
        raw_energies.clear()
        soft_weights.clear()

        del selected_clients_this_round
        del honest_clients
        del malicious_clients_this_round

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final Analysis and OPS Calculation will be done after the loop to ensure we have the full history for accurate OPS computation.
    if defense_key == "fedavg":
        FEDAVG_BASELINES[baseline_key] = {
            "round_acc": history["mta"].copy(),
            "round_asr": history["attack_ratio"].copy(),
        }

        with open(baseline_filepath, "w") as f:
            json.dump(FEDAVG_BASELINES[baseline_key], f)
        log_and_print(f"[INFO] Saved FedAvg baseline to cache: {baseline_filepath}", log_file_path=log_file_path)
    
    csv_filename = dynamic_csv_dir / f"acc_history_{defense_name}_alpha_{alpha}_attack_{attack_category}_ratio_{percent_attackers}_update_{update_method}_seed_{seed}_poison_{poison_ratio}.csv"
    
    # Final Analysis and OPS Calculation
    best_accuracy = max(history["mta"])
    best_round = history["mta"].index(best_accuracy) + 1
    last_accuracy = history["mta"][-1]

    valid_ops = [x for x in history["ops"] if not np.isnan(x)]
    valid_asr = [x for x in history["attack_ratio"] if not np.isnan(x)]

    mean_round_ops = float(np.mean(valid_ops)) if valid_ops else np.nan
    last_round_ops = float(history["ops"][-1]) if history["ops"] else np.nan

    mean_round_asr = float(np.mean(valid_asr)) if valid_asr else np.nan
    last_round_asr = float(history["attack_ratio"][-1]) if history["attack_ratio"] else np.nan

    pipeline_end_time = time.time()
    total_runtime = pipeline_end_time - pipeline_start_time

    OPS_RESULTS[(baseline_key, defense_key)] = {
        "best_accuracy": float(best_accuracy),
        "last_accuracy": float(last_accuracy),
        "mean_round_ops": mean_round_ops,
        "last_round_ops": last_round_ops,
        "mean_round_asr": mean_round_asr,
        "last_round_asr": last_round_asr,
        "total_runtime": total_runtime
    }
    
    df = pd.DataFrame({
        "Round": history["round"], 
        "Accuracy": history["mta"],
        "Precision": history["precision"],
        "Recall": history["recall"],
        "F1_Score": history["f1"],
        "Honest_Avg_Weight": history["honest_avg_weight"],
        "Malicious_Avg_Weight": history["malicious_avg_weight"],
        "TPR_Detection_Rate": history["tpr"],
        "FPR_False_Alarm_Rate": history["fpr"],
        "AUC": history["auc"],
        "AP": history["ap"],
        "TPR_at_5_FPR": history["tpr_5"],
        "TPR_at_10_FPR": history["tpr_10"],
        "Attack_Num": history["attack_num"],
        "Attack_Pass": history["attack_pass"],
        "Attack_Ratio": history["attack_ratio"],
        "FedAvg_Acc_Baseline": history["fedavg_acc_baseline"],
        "FedAvg_ASR_Baseline": history["fedavg_asr_baseline"],
        "OPS": history["ops"],
    })
    df.to_csv(csv_filename, index=False)
    
    if defense_name == "tsfl" and client_weight_rows:
        client_weight_filename = dynamic_csv_dir / (
            f"client_weights_by_round_{defense_name}_alpha_{alpha}_attack_{attack_category}_ratio_{percent_attackers}_update_{update_method}_seed_{seed}_poison_{poison_ratio}.csv"
        )

        df_client_weights = pd.DataFrame(client_weight_rows)

        df_client_weights["Round_display"] = df_client_weights["Round"].astype(str)

        df_client_weights.loc[
            df_client_weights["Round"].duplicated(),
            "Round_display"
        ] = ""

        df_client_weights = df_client_weights[
            ["Round_display", "Client_id", "energies", "soft_weights", "weights"]
        ]

        df_client_weights = df_client_weights.rename(columns={
            "Round_display": "Round"
        })

        df_client_weights.to_csv(client_weight_filename, index=False)

    if defense_name == "tsfl":
        msg = f"Generating Visualizations for Peak Round: {best_round_stats['round']} (Acc: {best_accuracy_so_far:.2f}%)"
        log_and_print(f"{msg}", log_file_path=log_file_path)
        
        visualizer = TSFLReport(
            graph_dir=dynamic_graph_dir,
            alpha=alpha,
            attack_category=attack_category,
            update_method=update_method,
            seed=seed,
            num_rounds=num_rounds,
            poison_ratio=poison_ratio,
            log_file_path=log_file_path
        )
        visualizer.generate_all_reports(history, best_round_stats, malicious_clients)

    msg = f"Final Global Accuracy for Alpha {alpha} with {defense_name} defense from attack category {attack_category}: {current_accuracy:.2f}%"


    msg_peak = f"Peak Global Accuracy: {best_accuracy:.2f}% (Achieved at Round {best_round})"
    msg_last = f"Converged Global Accuracy (Round {num_rounds}): {last_accuracy:.2f}%"
    msg_summary = f"[{defense_name.upper()}] Alpha {alpha} | Attack: {attack_category.upper()} -> Best: {best_accuracy:.2f}% | Last: {last_accuracy:.2f}%"
    msg_time = f"Total Runtime for {defense_name.upper()}: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)"
    
    # Calculate valid AUC/AP ignoring NaNs
    valid_auc = [x for x in history["auc"] if not np.isnan(x)]
    valid_ap = [x for x in history["ap"] if not np.isnan(x)]
    valid_tpr_5 = [x for x in history["tpr_5"] if not np.isnan(x)]
    valid_tpr_10 = [x for x in history["tpr_10"] if not np.isnan(x)]

    mean_auc = float(np.mean(valid_auc)) if valid_auc else np.nan
    mean_ap = float(np.mean(valid_ap)) if valid_ap else np.nan
    mean_tpr_5 = float(np.mean(valid_tpr_5)) if valid_tpr_5 else np.nan
    mean_tpr_10 = float(np.mean(valid_tpr_10)) if valid_tpr_10 else np.nan

    log_and_print("-" * 50, log_file_path=log_file_path)
    log_and_print(msg_peak, log_file_path=log_file_path)
    log_and_print(msg_last, log_file_path=log_file_path)
    log_and_print(msg_time, log_file_path=log_file_path)
    log_and_print("-" * 50, log_file_path=log_file_path)
    log_and_print(msg_summary, log_file_path=log_file_path)
    log_and_print(
        f"Mean Round ASR: {mean_round_asr:.2f} | Last Round ASR: {last_round_asr:.2f}",
        log_file_path=log_file_path
    )
    log_and_print(
        f"Mean Round OPS: {mean_round_ops:.2f} | Last Round OPS: {last_round_ops:.2f}",
        log_file_path=log_file_path
    )
    log_and_print(f"Mean AUC: {mean_auc:.4f}", log_file_path=log_file_path)
    log_and_print(f"Mean AP: {mean_ap:.4f}", log_file_path=log_file_path)
    log_and_print(f"Mean TPR@5%FPR: {mean_tpr_5:.2f}%", log_file_path=log_file_path)
    log_and_print(f"Mean TPR@10%FPR: {mean_tpr_10:.2f}%", log_file_path=log_file_path)
    return best_accuracy, total_runtime