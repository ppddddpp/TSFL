import numpy as np
from typing import Dict, List
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch.nn as nn

from tsfl.tsfl_helpers.tsfl_tensor_helper import flatten_client_updates_to_1d_numpy

class KNNGraphBuilder:
    def __init__(self, num_neighbors: int = 3):
        self.num_neighbors = num_neighbors

    def build(self, similarity_matrix: np.ndarray) -> np.ndarray:
        n_clients = similarity_matrix.shape[0]
        
        k = min(self.num_neighbors, n_clients - 1)
        if k <= 0:
            return np.zeros_like(similarity_matrix, dtype=bool)
        
        # Force the diagonal (self-similarity) to negative infinity for not to be selected
        similarity_matrix_no_self = similarity_matrix.copy()
        np.fill_diagonal(similarity_matrix_no_self, -np.inf)

        # Get K largest entries in each row (largest value lives in the last column)
        partition_idx = np.argpartition(similarity_matrix_no_self, -k, axis=1)
        
        # The last K columns are the top K neighbors
        top_k_indices = partition_idx[:, -k:]

        # Create adjacency matrix using advanced indexing
        adj = np.zeros_like(similarity_matrix_no_self, dtype=bool)
        rows = np.arange(n_clients)[:, None]
        adj[rows, top_k_indices] = True

        # Symmetrize (If A is neighbor of B, B is neighbor of A)
        adj_sym = np.logical_and(adj, adj.T)

        return adj_sym
    
class SybilDetector:
    def __init__(
            self,
            min_cluster_size: int = 2,
            num_neighbors: int = 3,
            epsilon_for_safe_normalization: float = 1e-8,

            graph_builder: KNNGraphBuilder = None,
        ):
        self.min_cluster_size = min_cluster_size
        self.epsilon_for_safe_normalization = epsilon_for_safe_normalization

        if graph_builder is not None:
            self.graph_builder = graph_builder
        else:
            self.graph_builder = KNNGraphBuilder(num_neighbors=num_neighbors)

    def detect_sybils(self, client_updates: Dict[str, dict], global_model: nn.Module, log_file_path: str = None) -> dict:
        # Flatten all client dictionaries into 1D arrays
        param_keys = []
        param_shapes = {}

        for k, v in global_model.named_parameters():
            param_keys.append(k)
            param_shapes[k] = v.shape
        
        flattened_dict = flatten_client_updates_to_1d_numpy(
            client_updates_need_to_flatten=client_updates, 
            param_keys=param_keys,
            param_shapes=param_shapes,
            log_file_path=log_file_path
        )
        ids = list(flattened_dict.keys())

        # Stack all update vectors into a matrix of shape (n, d) for n clients and d dimensions
        update_vectors_list = []
        for client_id in ids:
            update_vectors_list.append(flattened_dict[client_id])
        stacked_update_matrix = np.vstack(update_vectors_list)

        # Normalize for cosine similarity (which calls for L2 normalization)
        normalized_update_matrix = stacked_update_matrix / (np.linalg.norm(stacked_update_matrix, axis=1, keepdims=True) 
                                                            + self.epsilon_for_safe_normalization)

        # Cosine similarity
        similarity_matrix = np.dot(normalized_update_matrix, np.transpose(normalized_update_matrix))
        similarity_matrix_filled = np.nan_to_num(similarity_matrix, nan=0.0)

        # Build connectivity graph
        adjacency_matrix = self.graph_builder.build(similarity_matrix_filled)

        graph = csr_matrix(adjacency_matrix)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

        weighted_adj = similarity_matrix_filled * adjacency_matrix

        degrees = np.sum(adjacency_matrix, axis=1)
        strengths = np.sum(weighted_adj, axis=1)

        mean_neighbor_sim = strengths / (degrees + 1e-8)

        local_variances = []
        for i in range(len(ids)):
            neighbor_sims = similarity_matrix_filled[i][adjacency_matrix[i]]
            if len(neighbor_sims) > 0:
                local_variances.append(np.var(neighbor_sims))
            else:
                local_variances.append(0.0)

        local_variances = np.array(local_variances)


        # Assign each client to a cluster
        clusters = [[] for _ in range(n_components)]
        for client_idx, cluster_label in enumerate(labels):
            clusters[cluster_label].append(client_idx)
        
        cluster_features = {}
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) < self.min_cluster_size:
                continue
            
            # Analyze pairwise similarities within the cluster to compute robust similarity metrics
            submatrix = similarity_matrix_filled[np.ix_(cluster, cluster)]
            mask = ~np.eye(len(cluster), dtype=bool)
            pairwise_sims = submatrix[mask]

            # Compute robust statistics like 90th percentile, mean, and std of pairwise similarities to characterize the cluster's internal cohesion
            if len(pairwise_sims) > 0:
                robust_sim = np.percentile(pairwise_sims, 90)
                mean_sim = np.mean(pairwise_sims)
                std_sim = np.std(pairwise_sims)
            else:
                robust_sim = 0.0
                mean_sim = 0.0
                std_sim = 0.0

            # Normalize centroid properly
            cluster_vectors = stacked_update_matrix[cluster]
            centroid = np.mean(cluster_vectors, axis=0)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

            # Calculate similarity of each cluster member to the centroid to identify how tightly they align with the cluster's central tendency, 
            # which can help differentiate between structured Sybil groups and random clusters of honest clients.
            cluster_vectors_norm = cluster_vectors / (
                np.linalg.norm(cluster_vectors, axis=1, keepdims=True) + 1e-8
            )

            # The similarity to the centroid can help identify if the cluster is a tight Sybil group (high similarity to centroid)
            #  or a more diffuse cluster of honest clients (lower similarity to centroid).
            sims_to_centroid = np.dot(cluster_vectors_norm, centroid_norm)

            for local_idx, global_idx in enumerate(cluster):
                cid = ids[global_idx]

                cluster_features[cid] = {
                    "cluster_size": len(cluster),
                    "cluster_similarity_p90": float(robust_sim),
                    "cluster_similarity_mean": float(mean_sim),
                    "cluster_similarity_std": float(std_sim),
                    "centroid_similarity": float(sims_to_centroid[local_idx]),
                }

        client_scores = {}

        for i, client_id in enumerate(ids):
            sim_vals = similarity_matrix_filled[i][np.arange(len(ids)) != i]
            sorted_sims = np.sort(sim_vals)

            base_scores = {
                "max_similarity": float(sorted_sims[-1]),
                "mean_similarity": float(np.mean(sim_vals)),
                "similarity_variance": float(np.var(sim_vals)),
                "similarity_gap": float(sorted_sims[-1] - sorted_sims[-2]) if len(sorted_sims) > 1 else 0.0,    
                "degree": float(degrees[i]),
                "mean_neighbor_similarity": float(mean_neighbor_sim[i]),
                "local_similarity_variance": float(local_variances[i]),

                "cluster_similarity_p90": 0.0,
                "cluster_similarity_mean": 0.0,
                "cluster_similarity_std": 0.0,
                "centroid_similarity": 0.0,        # If isolated, they are their own centroid
            }

            if client_id in cluster_features:
                base_scores.update(cluster_features[client_id])

            client_scores[client_id] = base_scores

        out = {
            "keep_ids": ids,
            "client_scores": client_scores,

            "similarity_matrix": similarity_matrix_filled,
            "adjacency_matrix": adjacency_matrix,
            "connectivity_graph": graph,
            "labels": labels,
            "n_components": n_components,
            "clusters": clusters,
        }

        return out