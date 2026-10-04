import os
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from helpers import log_and_print
import seaborn as sns

class TSFLReport:
    def __init__(self, graph_dir: Path, alpha: float, attack_category: str, update_method: str, seed: int, num_rounds: int, poison_ratio: float, log_file_path=None):
        self.num_rounds = num_rounds
        self.log_file_path = log_file_path
        self.poison_ratio = poison_ratio

        # Define Prefix
        self.file_prefix = f"alpha_{alpha}_type_{attack_category}_update_{update_method}_seed_{seed}_poison_{poison_ratio}"
        
        self.graph_dir = graph_dir / self.file_prefix
        self.csv_dir = self.graph_dir
        
        os.makedirs(self.graph_dir, exist_ok=True)

    def plot_convergence(self, history: dict):
        log_and_print("\nGenerating Convergence Graph...", log_file_path=self.log_file_path)
        plt.figure(figsize=(8, 5))
        plt.plot(history["round"], history["mta"], marker='o', linestyle='-', color='b', label='TsflMain (Defended)')
        plt.title("Federated learning convergence curve for (CIFAR-10)")
        plt.xlabel("Communication round (T)")
        plt.ylabel("Main Task Accuracy (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        save_path = self.graph_dir / f"{self.file_prefix}_convergence.png"
        plt.savefig(save_path)
        plt.close()
        log_and_print(f"Graph saved at {save_path}.", log_file_path=self.log_file_path)

    def plot_energy_vs_weight(self, final_round_stats: dict, malicious_clients: list):
        log_and_print("\nGenerating Empirical Energy vs. Weight Plot...", log_file_path=self.log_file_path)
        plt.figure(figsize=(8, 5))

        honest_E, honest_W = [], []
        malicious_E, malicious_W = [], []

        for cid in final_round_stats.get("energies", {}).keys():
            e = final_round_stats["energies"][cid]
            w = final_round_stats["weights"][cid]
            if cid in malicious_clients:
                malicious_E.append(e)
                malicious_W.append(w)
            else:
                honest_E.append(e)
                honest_W.append(w)

        plt.scatter(honest_E, honest_W, color='#1f77b4', s=120, edgecolor='black', alpha=0.9, label='Honest clients (Actual)')
        plt.scatter(malicious_E, malicious_W, color='#d62728', marker='X', s=150, edgecolor='black', alpha=0.9, label='Malicious clients (Actual)')

        plt.title(f"Empirical defense behavior (Round {self.num_rounds} actual data)", fontsize=14, fontweight='bold')
        plt.xlabel(r"Actual True Energy ($E_i$)", fontsize=12)
        plt.ylabel(r"Assigned Aggregation Weight ($w_i$)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        save_path = self.graph_dir / f"{self.file_prefix}_energy_vs_weight.png"
        plt.savefig(save_path)
        plt.close()
        log_and_print(f"Graph saved at {save_path}.", log_file_path=self.log_file_path)

    def plot_behavioral_space(self, final_round_stats: dict, malicious_clients: list):
        log_and_print("\nGenerating empirical behavioral space plot...", log_file_path=self.log_file_path)
        plt.figure(figsize=(8, 6))

        honest_acc, honest_kl = [], []
        malicious_acc, malicious_kl = [], []

        if "dataset" in final_round_stats.get("val_out", {}):
            dataset_scores = final_round_stats["val_out"]["dataset"]["client_scores"]
            for cid, scores in dataset_scores.items():
                acc_delta = scores.get("reference_delta_acc_norm", 0.0)
                kl_div = scores.get("reference_kl_norm", 0.0)
                if cid in malicious_clients:
                    malicious_acc.append(acc_delta)
                    malicious_kl.append(kl_div)
                else:
                    honest_acc.append(acc_delta)
                    honest_kl.append(kl_div)

        plt.scatter(honest_acc, honest_kl, c='green', marker='o', s=60, alpha=0.7, edgecolors='k', label='Honest clients (Actual)')
        plt.scatter(malicious_acc, malicious_kl, c='red', marker='^', s=80, alpha=0.9, edgecolors='k', label='Malicious clients (Actual)')

        plt.title(f"Empirical behavioral feature space (Round {self.num_rounds})", fontsize=14, fontweight='bold')
        plt.xlabel(r"Actual accuracy delta ($\Delta Acc$)", fontsize=12)
        plt.ylabel(r"Actual KL divergence ($D_{KL}$)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        
        save_path = self.graph_dir / f"{self.file_prefix}_behavioral_space.png"
        plt.savefig(save_path)
        plt.close()
        log_and_print(f"Graph saved at {save_path}.", log_file_path=self.log_file_path)

    def plot_weight_evolution(self, history: dict):
        log_and_print("\nGenerating weight evolution Graph...", log_file_path=self.log_file_path)
        plt.figure(figsize=(8, 5))
        plt.plot(history["round"], history["honest_avg_weight"], label="Avg honest client weight", color="green", linewidth=2)
        plt.plot(history["round"], history["malicious_avg_weight"], label="Avg malicious client weight", color="red", linestyle="--", linewidth=2)
        
        plt.title("Evolution of Aggregation weights over Communication rounds", fontsize=14, fontweight="bold")
        plt.xlabel("Communication rounds (T)", fontsize=12)
        plt.ylabel("Assigned softmax weight ($w_i$)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        save_path = self.graph_dir / f"{self.file_prefix}_weight_evolution.png"
        plt.savefig(save_path)
        plt.close()
        log_and_print(f"Graph saved at {save_path}.", log_file_path=self.log_file_path)

    def save_history_csv(self, history: dict):
        log_and_print(f"\nSaving simulation history to CSV...", log_file_path=self.log_file_path)
        csv_file_path = self.csv_dir / f"{self.file_prefix}_history.csv"
        
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["round", "mta_accuracy", "honest_avg_weight", "malicious_avg_weight"])
            for i in range(len(history["round"])):
                writer.writerow([
                    history["round"][i],
                    history["mta"][i],
                    history["honest_avg_weight"][i],
                    history["malicious_avg_weight"][i]
                ])
                
        log_and_print(f"Data saved successfully at {csv_file_path}.", log_file_path=self.log_file_path)

    def plot_energy_density(self, final_round_stats: dict, malicious_clients: list):
        log_and_print("\nGenerating Energy Density (KDE) Plot...", log_file_path=self.log_file_path)
        plt.figure(figsize=(8, 5))

        honest_E = [final_round_stats["energies"][cid] for cid in final_round_stats["energies"] if cid not in malicious_clients]
        malicious_E = [final_round_stats["energies"][cid] for cid in final_round_stats["energies"] if cid in malicious_clients]

        sns.kdeplot(honest_E, fill=True, color='#1f77b4', label='Honest Clients', linewidth=2)
        if malicious_E:
            sns.kdeplot(malicious_E, fill=True, color='#d62728', label='Malicious Clients', linewidth=2)

        plt.title(f"Energy Density Distribution (Round {final_round_stats.get('round', self.num_rounds)})", fontsize=14, fontweight='bold')
        plt.xlabel(r"Raw Anomaly Energy ($E_i$)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        
        if honest_E:
            plt.axvline(x=max(honest_E), color='gray', linestyle='--', label='Inlier Max Boundary')
        else:
            print(f"[WARNING] honest_E is empty for this case. Skipping Inlier Max Boundary plot.")
        
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        save_path = self.graph_dir / f"{self.file_prefix}_energy_density.png"
        plt.savefig(save_path)
        plt.close()
        log_and_print(f"Graph saved at {save_path}.", log_file_path=self.log_file_path)

    def plot_weight_distribution(self, final_round_stats: dict, malicious_clients: list):
        log_and_print("\nGenerating Sorted Weight Distribution Plot...", log_file_path=self.log_file_path)
        plt.figure(figsize=(10, 5))

        weights_dict = final_round_stats["weights"]
        sorted_clients = sorted(weights_dict.keys(), key=lambda k: weights_dict[k], reverse=True)
        
        sorted_weights = [weights_dict[cid] for cid in sorted_clients]
        colors = ['#d62728' if cid in malicious_clients else '#1f77b4' for cid in sorted_clients]
        labels = ['Malicious' if cid in malicious_clients else 'Honest' for cid in sorted_clients]
        bars = plt.bar(range(len(sorted_weights)), sorted_weights, color=colors, edgecolor='black', alpha=0.8)

        plt.title(f"Egalitarian Weight Distribution (Round {final_round_stats.get('round', self.num_rounds)})", fontsize=14, fontweight='bold')
        plt.xlabel("Clients (Sorted by Assigned Weight)", fontsize=12)
        plt.ylabel(r"Assigned Softmax Weight ($w_i$)", fontsize=12)
        
        # Custom Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', edgecolor='black', label='Honest Clients'),
                           Patch(facecolor='#d62728', edgecolor='black', label='Malicious Clients')]
        plt.legend(handles=legend_elements, fontsize=11)
        
        plt.xticks([])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_path = self.graph_dir / f"{self.file_prefix}_weight_distribution.png"
        plt.savefig(save_path)
        plt.close()
        log_and_print(f"Graph saved at {save_path}.", log_file_path=self.log_file_path)

    def plot_weighting_ablation(self, final_round_stats: dict, malicious_clients: list):
        log_and_print("\nGenerating Weighting Function Ablation Plot...", log_file_path=self.log_file_path)
        plt.figure(figsize=(9, 6))

        energies_dict = final_round_stats["energies"]
        cids = list(energies_dict.keys())
        raw_E = np.array([energies_dict[cid] for cid in cids])
        
        # Sort clients by their raw energy for better visualization
        sort_indices = np.argsort(raw_E)
        sorted_E = raw_E[sort_indices]
        sorted_cids = [cids[i] for i in sort_indices]

        #  Inverse Proportional (W = 1 / E)
        inv_W = 1.0 / (sorted_E + 1e-6)
        inv_W = inv_W / np.sum(inv_W) # L1 Norm

        # Standard Softmax (W = exp(-E))
        norm_E = (sorted_E - np.min(sorted_E)) / (np.std(sorted_E) + 1e-6)
        soft_W = np.exp(-norm_E)
        soft_W = soft_W / np.sum(soft_W) # L1 Norm

        # TSFL Proposed 
        tsfl_W = np.array([final_round_stats["weights"][cid] for cid in sorted_cids])

        colors = ['#d62728' if cid in malicious_clients else '#1f77b4' for cid in sorted_cids]

        plt.plot(sorted_E, inv_W * 100, linestyle='--', color='gray', linewidth=2, label=r'Inverse Mapping ($w \propto 1/E$) - High Bias')
        plt.plot(sorted_E, soft_W * 100, linestyle='-.', color='orange', linewidth=2, label=r'Standard Softmax ($w \propto \exp(-E)$) - High Bias')
        plt.plot(sorted_E, tsfl_W * 100, linestyle='-', color='green', linewidth=3, label='TSFL Pure Egalitarian (Proposed)')
        
        plt.scatter(sorted_E, tsfl_W * 100, color=colors, s=80, edgecolor='black', zorder=5)

        plt.title(f"Ablation of Weighting Functions (Round {final_round_stats.get('round', self.num_rounds)})", fontsize=14, fontweight='bold')
        plt.xlabel(r"Raw Anomaly Energy ($E_i$)", fontsize=12)
        plt.ylabel("Assigned Normalized Weight (%)", fontsize=12)
        
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        save_path = self.graph_dir / f"{self.file_prefix}_weighting_ablation.png"
        plt.savefig(save_path)
        plt.close()
        log_and_print(f"Graph saved at {save_path}.", log_file_path=self.log_file_path)

    def generate_all_reports(self, history: dict, final_round_stats: dict, malicious_clients: list):
        """Wrapper to generate all plots and CSVs in one clean call."""
        log_and_print("\n--- Generating Simulation Reports ---", log_file_path=self.log_file_path)
        self.plot_convergence(history)
        self.plot_energy_vs_weight(final_round_stats, malicious_clients)
        self.plot_behavioral_space(final_round_stats, malicious_clients)
        self.plot_weight_evolution(history)
        self.plot_energy_density(final_round_stats, malicious_clients)
        self.plot_weight_distribution(final_round_stats, malicious_clients)
        self.save_history_csv(history)
        self.plot_weighting_ablation(final_round_stats, malicious_clients)
        log_and_print("--- Reporting Complete ---\n", log_file_path=self.log_file_path)