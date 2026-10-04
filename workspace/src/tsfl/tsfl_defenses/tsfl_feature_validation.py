from typing import Literal
from torch.utils.data import DataLoader

from .sub_module import (
    SybilDetector,
    UpdateScaleValidator,
    UpdateDirectionEvaluator,
    DatasetEvaluator,
)

class TsflFeatureValidator:
    def __init__(
            self,

            # Toggle parameters
            use_sybil_detector: bool = True,
            use_update_scale_validator: bool = True,
            use_update_direction_evaluator: bool = True,
            use_dataset_evaluator: bool = True,

            # Sybil detector parameters
            min_cluster_size: int = 2,
            num_neighbors: int = 3,
            epsilon_for_safe_normalization: float = 1e-8,

            # Scale validation parameters
            tolerance_multiplier: float = 3.0,
            fallback_threshold: float = 1e-5,

            # Direction validation parameters
            aggregation_method: Literal["mean", "median"] = "median", 
            eps: float = 1e-8,
            lower_scale_mad_for_cosine: float = 0.05,

            # Dataset evaluator parameters
            dataset_loader: DataLoader = None,
            max_samples: int = 512,
            T_kl: float = 2.0,
            seed: int = 2709
        ):
        self.loaded_params = {}

        if use_sybil_detector:
            self.sybil_detector = SybilDetector(
                min_cluster_size=min_cluster_size,
                num_neighbors=num_neighbors,
                epsilon_for_safe_normalization=epsilon_for_safe_normalization
            )

            self.loaded_params["sybil_detector"] = {
                "min_cluster_size": min_cluster_size,
                "num_neighbors": num_neighbors,
                "epsilon_for_safe_normalization": epsilon_for_safe_normalization
            }
        else:
            self.sybil_detector = None

        if use_update_scale_validator:
            self.update_scale_validator = UpdateScaleValidator(
                tolerance_multiplier=tolerance_multiplier,
                fallback_threshold=fallback_threshold
            )

            self.loaded_params["update_scale_validator"] = {
                "tolerance_multiplier": tolerance_multiplier,
                "fallback_threshold": fallback_threshold
            }
        else:
            self.update_scale_validator = None

        if use_update_direction_evaluator:
            self.update_direction_evaluator = UpdateDirectionEvaluator(
                aggregation_method=aggregation_method,
                eps=eps,
                lower_scale_mad_for_cosine=lower_scale_mad_for_cosine
            )

            self.loaded_params["update_direction_evaluator"] = {
                "aggregation_method": aggregation_method,
                "eps": eps,
                "lower_scale_mad_for_cosine": lower_scale_mad_for_cosine
            }
        else:
            self.update_direction_evaluator = None

        if use_dataset_evaluator:
            self.dataset_evaluator = DatasetEvaluator(
                dataset_loader=dataset_loader,
                max_samples=max_samples,
                T_kl=T_kl,
                seed=seed
            )

            self.loaded_params["dataset_evaluator"] = {
                "max_samples": max_samples,
                "T_kl": T_kl,
            }
        else:
            self.dataset_evaluator = None

        self.use_sybil_detector = use_sybil_detector
        self.use_update_scale_validator = use_update_scale_validator
        self.use_update_direction_evaluator = use_update_direction_evaluator
        self.use_dataset_evaluator = use_dataset_evaluator

    def evaluate_sybil(self, client_updates, global_model, log_file_path=None)->dict:
        if client_updates is None:
            raise ValueError("client_updates cannot be None")
        return self.sybil_detector.detect_sybils(client_updates=client_updates, global_model=global_model, log_file_path=log_file_path)

    def evaluate_update_scale(self, client_updates, global_model, log_file_path=None)->dict:
        if client_updates is None:
            raise ValueError("client_updates cannot be None")
        return self.update_scale_validator.validate_scales(client_updates=client_updates, global_model=global_model, log_file_path=log_file_path)

    def evaluate_update_direction(self, client_updates, global_model, log_file_path=None)->dict:
        if client_updates is None:
            raise ValueError("client_updates cannot be None")
        return self.update_direction_evaluator.validate_directions(client_updates=client_updates, global_model=global_model, log_file_path=log_file_path)
    
    def evaluate_dataset(self, global_model, client_updates, device=None)->dict:
        if client_updates is None:
            raise ValueError("client_updates cannot be None")
        return self.dataset_evaluator.evaluate_on_dataset(global_model=global_model, client_updates=client_updates, device=device)