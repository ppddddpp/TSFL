from .fedavg import FedAvg
from .lasa import LASA
from .signguard import SignGuard
from .foolsgold import FoolsGold
from .feddlad import FedDLAD
from .fltrust import FLTrustDefense

import torch

class OtherDefensesSetup:
    @staticmethod
    def get_built_in_defenses():
        return [
            'tsfl', # Default for calling
            'fedavg',
            'lasa',
            'signguard',
            'foolsgold',
            'feddlad',
            'fltrust'
        ]

    @staticmethod
    def define_defenses(
        defense_name: str, 
        dataset_name: str, 
        clients_per_round: int = 25, 
        server_proxy_loader=None
    ):
        name = defense_name.lower().strip()

        if name == 'fedavg':
            return FedAvg()

        if name == 'lasa':
            lambda_val = 1.0 if dataset_name in ['cifar10', 'cifar100', 'cifar', 'noniidcifar', 'noniidcifar100'] else 2.0
            return LASA(
                num_selected_users=clients_per_round,
                sparsity=0.3,
                lambda_n=lambda_val,
                lambda_s=1.0,
            )

        if name == "signguard":
            return SignGuard(
                num_selected_users=clients_per_round
            )

        if name == "foolsgold":
            return FoolsGold()

        if name == "feddlad":
            return FedDLAD(
                bg=12,
                pg=3,
                iqr_scale=0.6,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        if name == "fltrust":
            return FLTrustDefense(
                server_proxy_loader=server_proxy_loader,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

        raise ValueError(f"Defense '{defense_name}' is not defined or supported.")