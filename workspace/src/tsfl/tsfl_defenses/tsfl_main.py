import numpy as np

from .custom_tree import CustomIsolationForest
from .tsfl_feature_validation import TsflFeatureValidator
from tsfl.tsfl_helpers import WeightUpdater
from helpers import log_and_print
import torch

class TsflMain:
    def __init__(
            self,
            custom_isolation_forest=None,
            tsfl_feature_validator=None,
            weight_updater=None
        ):
        self.custom_isolation_forest = CustomIsolationForest() if custom_isolation_forest is None else custom_isolation_forest
        self.tsfl_feature_validator = TsflFeatureValidator() if tsfl_feature_validator is None else tsfl_feature_validator
        self.weight_updater = WeightUpdater() if weight_updater is None else weight_updater

        self.validation_output = {}
    
    def _build_feature_matrix(self, validation_output, surviving_ids):
        feature_matrix = {cid: {} for cid in surviving_ids}

        for cid in surviving_ids:
            # 1. Structural features
            if "sybil" in validation_output:
                scores = validation_output["sybil"]["client_scores"].get(cid, {})
                
                feature_matrix[cid]["cluster_size"]              = scores.get("cluster_size", 1)
                feature_matrix[cid]["s_p90"]                     = scores.get("cluster_similarity_p90", 0.0)
                feature_matrix[cid]["centroid_sim"]              = scores.get("centroid_similarity", 1.0)
                feature_matrix[cid]["max_sim"]                   = scores.get("max_similarity", 0.0)
                feature_matrix[cid]["sim_variance"]              = scores.get("similarity_variance", 0.0)
                feature_matrix[cid]["mean_sim"]                  = scores.get("mean_similarity", 0.0)
                feature_matrix[cid]["sim_gap"]                   = scores.get("similarity_gap", 0.0)
                feature_matrix[cid]["degree"]                    = scores.get("degree", 0.0)
                feature_matrix[cid]["mean_neighbor_sim"]         = scores.get("mean_neighbor_similarity", 0.0)
                feature_matrix[cid]["local_sim_var"]             = scores.get("local_similarity_variance", 0.0)
                feature_matrix[cid]["cluster_similarity_std"]    = scores.get("cluster_similarity_std", 0.0)
                feature_matrix[cid]["cluster_similarity_mean"]   = scores.get("cluster_similarity_mean", 0.0)

                c_size = scores.get("cluster_size", 1)
                sim_var = scores.get("similarity_variance", 0.0)
                max_sim = scores.get("max_similarity", 0.0)
                if c_size > 2 and max_sim > 0.90:
                    collusion_penalty = np.log1p(c_size) / (sim_var + 1e-4)
                else:
                    collusion_penalty = 0.0
                    
                feature_matrix[cid]["collusion_penalty"] = collusion_penalty

            # 2. Magnitude features (Scale validation)
            if "update_scale" in validation_output:
                scores = validation_output["update_scale"]["client_scores"].get(cid, {})

                feature_matrix[cid]["scale_zscore"]    = scores.get("scale_zscore", 0.0)
                feature_matrix[cid]["scale_anomaly"]   = scores.get("scale_anomaly_score", 0.0)

            # 3. Directional features (Direction validation)
            if "update_direction" in validation_output:
                scores = validation_output["update_direction"]["client_scores"].get(cid, {})
                
                feature_matrix[cid]["dir_zscore"]    = scores.get("global_deviation_z_score", 0.0)
                feature_matrix[cid]["dir_anomaly"]   = scores.get("direction_anomaly_score", 0.0)
                feature_matrix[cid]["dir_rank"]      = scores.get("cosine_rank", 0.5)
                feature_matrix[cid]["sign_ratio"]    = scores.get("sign_disagreement_ratio", 0.0)
                feature_matrix[cid]["sign_anomaly"]  = scores.get("sign_anomaly_score", 0.0)

            # 4. Behavioral features (Dataset validation)
            if "dataset" in validation_output:
                scores = validation_output["dataset"]["client_scores"].get(cid, {})
                feature_matrix[cid]["delta_acc"] = scores.get("reference_delta_acc_norm", 0.0)
                feature_matrix[cid]["delta_loss"] = scores.get("reference_delta_loss_norm", 0.0)
                feature_matrix[cid]["kl_div"] = scores.get("reference_kl_norm", 0.0)
                feature_matrix[cid]["kl_classwise"] = scores.get("reference_classwise_kl_norm", 0.0)
        
        return feature_matrix

    def _energy_guided_norm_bounding(self, client_updates, raw_energies, log_file_path):
        energies_array = np.array(list(raw_energies.values()))
        
        # Calcuate sigma using robust statistics to find the trusted subset
        median_e = float(np.median(energies_array))
        mad_e = float(np.median(np.abs(energies_array - median_e)))
        sigma = 3.0 * max(mad_e, 1e-6) * 1.4826

        # Get the trusted subset of clients whose energy is within the median + 3*sigma threshold
        trusted_cids = [cid for cid, energy in raw_energies.items() if energy <= (median_e + sigma)]
        
        # Find trust client 
        if len(trusted_cids) == 0:
            # If no trusted client is found, use 50% of clients with the lowest energy as a fallback
            sorted_cids = sorted(raw_energies.keys(), key=lambda k: raw_energies[k])
            trusted_cids = sorted_cids[:max(1, len(client_updates)//2)]

        client_norms = {}
        for cid, update_dict in client_updates.items():
            total_norm = 0.0
            for param in update_dict.values():
                total_norm += param.float().norm(2).item() ** 2
            client_norms[cid] = total_norm ** 0.5

        # Median norm of the trusted clients
        trusted_norms = [client_norms[cid] for cid in trusted_cids]
        median_trusted_norm = float(np.median(trusted_norms))

        # Clip the norms of all clients to be at most the median norm of the trusted clients
        clipped_updates = {}
        for cid, update_dict in client_updates.items():
            norm = client_norms[cid]
            # If norm is greater than the median trusted norm, scale it down to the median trusted norm
            # If norm is less than or equal to the median trusted norm, keep it as is (clip_factor = 1.0)
            clip_factor = min(1.0, median_trusted_norm / (norm + 1e-6))
            
            clipped_updates[cid] = {k: v * clip_factor for k, v in update_dict.items()}

        log_str = f"Norm Bounding | Trusted subset size: {len(trusted_cids)} | Safe Median Norm: {median_trusted_norm:.4f}\n"
        log_and_print(log_str, log_file_path=log_file_path)

        return clipped_updates

    def analyze(self, global_model, client_updates, log_file_path=None, device="cpu",
                lower_bound: float=1e-5, upper_bound: float=1.0)-> tuple:
        safe_client_updates = {}
        for cid, update in client_updates.items(): 
            is_valid = True
            for k, param in update.items():
                if not torch.isfinite(param).all():
                    is_valid = False
                    break
            
            if is_valid:
                safe_client_updates[cid] = update
            else:
                log_str = f"  [TSFL] Client {cid} update contains NaN or Inf values. Excluding from aggregation."
                log_and_print(log_str, log_file_path=log_file_path)
                
        client_updates = safe_client_updates
        
        if len(client_updates) == 0:
            log_str = "  [TSFL WARNING] No valid client updates found. Returning global model."
            log_and_print(log_str, log_file_path=log_file_path)
            return global_model, {}, {}, {}, {} 
        
        all_client_ids = list(client_updates.keys())
        self.validation_output = {}

        if self.tsfl_feature_validator.use_sybil_detector:
            self.validation_output["sybil"] = self.tsfl_feature_validator.evaluate_sybil(client_updates, global_model, log_file_path)

        if self.tsfl_feature_validator.use_update_scale_validator:
            self.validation_output["update_scale"] = self.tsfl_feature_validator.evaluate_update_scale(client_updates, global_model, log_file_path)

        if self.tsfl_feature_validator.use_update_direction_evaluator:
            self.validation_output["update_direction"] = self.tsfl_feature_validator.evaluate_update_direction(client_updates, global_model, log_file_path)

        if self.tsfl_feature_validator.use_dataset_evaluator:
            self.validation_output["dataset"] = self.tsfl_feature_validator.evaluate_dataset(global_model, client_updates, device)

        # Build the feature matrix for the Isolation Forest based on the validation outputs
        feature_matrix = self._build_feature_matrix(
            self.validation_output, 
            all_client_ids
        )
        
        # Compute the adaptive aggregation weights
        final_weights, raw_energies, soft_weights = self.custom_isolation_forest.compute_aggregation_weights(feature_matrix, lower_bound, upper_bound)

        client_updates = self._energy_guided_norm_bounding(client_updates, raw_energies, log_file_path)

        weight_log_str = "\n--- Calculated Soft Weights ---\n"
        for cid, weight in sorted(final_weights.items()):
            # Format the weight to 6 decimal places for clean alignment
            weight_log_str += f"{cid}: {weight:.6f}\n"
        weight_log_str += "-------------------------------"
        
        log_and_print(weight_log_str, log_file_path=log_file_path)

        # Apply the soft weights to the client updates
        return client_updates, self.validation_output, final_weights, raw_energies, soft_weights
    
    def update_weights(self, update_method, global_model, client_updates, soft_weights):
        new_state = self.weight_updater.apply_update_method(
            update_method=update_method,
            global_state=global_model.state_dict(),
            client_deltas=client_updates,
            client_weights=soft_weights
        )
        
        global_model.load_state_dict(new_state)
        return global_model