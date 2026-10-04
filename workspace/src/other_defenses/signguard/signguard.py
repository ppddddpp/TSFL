"""
This file is adapted for this project based on the algorithmic ideas of
SignGuard and LASA.

The SignGuard filtering strategy is inspired by:
[1] J. Xu, S.-L. Huang, L. Song, and T. Lan,
    "Byzantine-Robust Federated Learning through Collaborative Malicious
    Gradient Filtering," in Proc. IEEE ICDCS, pp. 1223-1235, 2022.

The implementation style and experimental defense setting are adapted from:
[2] J. Xu, Z. Zhang, and R. Hu,
    "Achieving Byzantine-Resilient Federated Learning via Layer-Adaptive
    Sparsified Model Aggregation," in Proc. IEEE/CVF WACV, pp. 1508-1517, 2025.

Original code references:
- SignGuard official implementation:
  https://github.com/JianXu95/SignGuard/tree/main/aggregators
- LASA official implementation:
  https://github.com/JiiahaoXU/LASA/tree/master/algorithms/defense

Thanks to the original authors for their contributions.

Modified by ppdddd et al. for integration with this project framework.
"""

import numpy as np
import torch

from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth

from other_defenses.utils.aggregation import (
    add_update_to_model,
    filter_finite_updates,
    get_update_keys,
    mean_update,
    stack_updates,
    vector_to_update,
)


class SignGuard:
    """SignGuard aggregation following LASA-style implementation."""

    def __init__(self, num_selected_users: int = 25):
        self.num_selected_users = int(num_selected_users)
        self.selected_indices = []
        self.norm_malicious_indices = []
        self.sign_malicious_indices = []

    def _aggregate(self, updates, keys):
        stacked, keys = stack_updates(updates, keys)

        num_users = len(updates)
        all_set = set([i for i in range(num_users)])
        iters = 1

        grads = stacked

        grad_l2norm = torch.norm(grads, dim=1).cpu().numpy()

        if np.any(np.isnan(grad_l2norm)):
            grad_l2norm = np.where(np.isnan(grad_l2norm), 0, grad_l2norm)

        norm_max = grad_l2norm.max()
        norm_med = np.median(grad_l2norm)

        benign_idx1 = all_set

        # Small adjustment to the original author's norm-based filtering strategy:
        # Flatten the gradients for computing the L2 norms 
        # for avoid "only 0-dimensional arrays can be converted to Python scalars" error
        benign_idx1 = benign_idx1.intersection(
            set([int(i) for i in np.argwhere(grad_l2norm > 0.1 * norm_med).flatten()]) 
        )
        benign_idx1 = benign_idx1.intersection(
            set([int(i) for i in np.argwhere(grad_l2norm < 3.0 * norm_med).flatten()])
        )

        num_param = grads.shape[1]
        num_spars = int(0.1 * num_param)

        benign_idx2 = all_set

        dbscan = 0

        for _ in range(iters):
            idx = torch.randint(
                0,
                (num_param - num_spars),
                size=(1,)
            ).item()

            gradss = grads[:, idx:(idx + num_spars)]

            sign_grads = torch.sign(gradss)

            sign_pos = (
                sign_grads.eq(1.0).sum(dim=1, dtype=torch.float32)
                / num_spars
            )
            sign_zero = (
                sign_grads.eq(0.0).sum(dim=1, dtype=torch.float32)
                / num_spars
            )
            sign_neg = (
                sign_grads.eq(-1.0).sum(dim=1, dtype=torch.float32)
                / num_spars
            )

            pos_max = sign_pos.max()
            pos_feat = sign_pos / (pos_max + 1e-8)

            zero_max = sign_zero.max()
            zero_feat = sign_zero / (zero_max + 1e-8)

            neg_max = sign_neg.max()
            neg_feat = sign_neg / (neg_max + 1e-8)

            feat = [pos_feat, zero_feat, neg_feat]
            sign_feat = torch.stack(feat, dim=1).cpu().numpy()

            if dbscan:
                clf_sign = DBSCAN(eps=0.05, min_samples=2).fit(sign_feat)
                labels = clf_sign.labels_

                n_cluster = len(set(labels)) - (1 if -1 in labels else 0)

                num_class = []
                for i in range(n_cluster):
                    num_class.append(np.sum(labels == i))

                benign_class = np.argmax(num_class)

                # Small adjustment to the original author's norm-based filtering strategy:
                # Flatten the gradients for computing the L2 norms 
                # for avoid "only 0-dimensional arrays can be converted to Python scalars" error
                benign_idx2 = benign_idx2.intersection(
                    set([int(i) for i in np.argwhere(labels == benign_class).flatten()])
                )

            else:
                bandwidth = estimate_bandwidth(
                    sign_feat,
                    quantile=0.5,
                    n_samples=num_users
                )

                # Small adjustment to the original author's norm-based filtering strategy:
                # Set bandwidth to 1e-5 if it is 0
                # for avoid "divide by zero" error
                if bandwidth  <= 0.0:
                    bandwidth  = 1e-5

                ms = MeanShift(
                    bandwidth=bandwidth,
                    bin_seeding=True,
                    cluster_all=False
                )

                ms.fit(sign_feat)

                labels = ms.labels_
                cluster_centers = ms.cluster_centers_

                labels_unique = np.unique(labels)
                n_cluster = len(labels_unique) - (1 if -1 in labels_unique else 0)

                num_class = []
                for i in range(n_cluster):
                    num_class.append(np.sum(labels == i))

                benign_class = np.argmax(num_class)

                # Small adjustment to the original author's norm-based filtering strategy:
                # Flatten the gradients for computing the L2 norms 
                # for avoid "only 0-dimensional arrays can be converted to Python scalars" error
                benign_idx2 = benign_idx2.intersection(
                    set([int(i) for i in np.argwhere(labels == benign_class).flatten()])
                )

        self.norm_malicious_indices = list(all_set - benign_idx1)
        self.sign_malicious_indices = list(all_set - benign_idx2)

        benign_idx = list(benign_idx2.intersection(benign_idx1))
        self.selected_indices = benign_idx

        grad_norm = torch.norm(grads, dim=1).reshape((-1, 1))
        norm_clip = grad_norm.median(dim=0)[0].item()
        grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
        grads_clip = (grads / grad_norm) * grad_norm_clipped

        global_grad = grads[benign_idx].mean(dim=0)

        return global_grad, keys

    def run(self, global_model, client_updates_dict):
        if not client_updates_dict:
            return global_model

        updates = list(client_updates_dict.values())
        keys = get_update_keys(updates, global_model)

        updates = filter_finite_updates(updates, keys)

        if not updates:
            return global_model

        global_grad, keys = self._aggregate(updates, keys)

        new_state = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in global_model.state_dict().items()
        }

        pointer = 0

        for key in keys:
            template = updates[0][key]

            num_param = template.numel()

            delta = global_grad[pointer:pointer + num_param].view_as(template)
            pointer += num_param

            if (
                key in new_state
                and torch.is_tensor(new_state[key])
                and new_state[key].shape == delta.shape
                and torch.is_floating_point(new_state[key])
            ):
                new_state[key] = new_state[key] + delta.to(
                    device=new_state[key].device,
                    dtype=new_state[key].dtype
                )

        global_model.load_state_dict(new_state, strict=True)

        return global_model