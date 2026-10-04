"""
This file is adapted for this project based on the algorithmic idea of
Federated Averaging (FedAvg).

The FedAvg aggregation strategy is inspired by:
[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas,
    "Communication-Efficient Learning of Deep Networks from Decentralized Data,"
    in Proc. International Conference on Artificial Intelligence and Statistics
    (AISTATS), 2017.

The implementation style and experimental defense setting are adapted from:
[2] J. Xu, Z. Zhang, and R. Hu,
    "Achieving Byzantine-Resilient Federated Learning via Layer-Adaptive
    Sparsified Model Aggregation," in Proc. IEEE/CVF WACV, 2025.

Original code references:
- LASA official implementation:
  https://github.com/JiiahaoXU/LASA/tree/master/algorithms/defense

Thanks to the original authors for their contributions.

Modified by ppdddd et al. for integration with this project framework.
"""
from other_defenses.utils.aggregation import add_update_to_model, filter_finite_updates, get_update_keys, mean_update


class FedAvg:
    """Non-robust baseline: average all client updates."""

    def __init__(self):
        pass

    def run(self, global_model, client_updates_dict):
        if not client_updates_dict:
            return global_model
        updates = list(client_updates_dict.values())
        keys = get_update_keys(updates, global_model)
        updates = filter_finite_updates(updates, keys)
        if not updates:
            return global_model
        aggregate = mean_update(updates, keys)
        return add_update_to_model(global_model, aggregate)
