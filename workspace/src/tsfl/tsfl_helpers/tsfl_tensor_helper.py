import numpy as np
import torch

from helpers import log_and_print

def flatten_client_updates_to_1d_numpy(client_updates_need_to_flatten: dict, log_file_path: str = None,
                                    param_keys: set = None, param_shapes: dict = None
                                    ) -> dict:
    flat_client_vectors = {}
    sorted_keys = sorted(param_keys)
    for client_id, layer_with_updates_dict in client_updates_need_to_flatten.items():
        
        # Check if layer_with_updates_dict is a dictionary of layer name to update
        if isinstance(layer_with_updates_dict, dict):
            flattened_layers = []

            # Iterate through the expected parameter keys in a consistent order
            for k in sorted_keys:
                if k in layer_with_updates_dict:
                    new_weight = layer_with_updates_dict[k]
                    if isinstance(new_weight, torch.Tensor):
                        new_weight_np = new_weight.detach().cpu().numpy().astype(np.float32)
                    else:
                        new_weight_np = np.array(new_weight)
                    expected_shape = param_shapes[k]

                    # Verify the shape of the new weight against the expected shape
                    if new_weight_np.shape != tuple(expected_shape):
                        log_and_print(
                            f"[WARNING] Shape mismatch for {k} in client {client_id}. "
                            f"Expected {expected_shape}, got {new_weight_np.shape}. Using zeros.",
                            log_file_path
                        )
                        new_weight_np = np.zeros(expected_shape, dtype=np.float32)

                    new_weight_np = new_weight_np.flatten()
                else:
                    shape = param_shapes[k]

                    # If the parameter is missing, fill with zeros of the correct shape and dtype
                    new_weight_np = np.zeros(shape, dtype=np.float32).flatten()
                    log_and_print(f"Parameter '{k}' missing in client '{client_id}' updates. Filling with zeros.", log_file_path=log_file_path)

                # Append the flattened layer to the list of layers for this client
                flattened_layers.append(new_weight_np)

            try:
                flattened_vector = np.concatenate(flattened_layers).astype(np.float32) if flattened_layers else np.array([0.0])

                # Safely store the vector mapped to its specific client ID
                flat_client_vectors[client_id] = flattened_vector
            except ValueError:
                log_and_print(f"Unable to concatenate layer updates for {client_id}. Skipping.", log_file_path=log_file_path)
                continue
        else:
            log_and_print(f"Unable to flatten layer updates for client {client_id}. Skipping this client.", log_file_path=log_file_path)
            continue

    return flat_client_vectors