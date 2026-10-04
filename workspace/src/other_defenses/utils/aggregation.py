from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch

StateDict = Dict[str, torch.Tensor]


def _is_usable_tensor(key: str, value: torch.Tensor) -> bool:
    if not torch.is_tensor(value):
        return False
    if 'num_batches_tracked' in key:
        return False
    if not torch.is_floating_point(value):
        return False
    return True


def get_update_keys(updates: Sequence[StateDict], global_model=None) -> List[str]:
    """Return common floating tensor keys that are safe to aggregate."""
    if not updates:
        return []
    base = updates[0]
    keys = []
    global_state = global_model.state_dict() if global_model is not None else None
    for key, value in base.items():
        if not _is_usable_tensor(key, value):
            continue
        if global_state is not None:
            if key not in global_state or global_state[key].shape != value.shape:
                continue
        ok = True
        for upd in updates[1:]:
            if key not in upd or not torch.is_tensor(upd[key]) or upd[key].shape != value.shape:
                ok = False
                break
        if ok:
            keys.append(key)
    return keys


def flatten_update(update: StateDict, keys: Optional[Sequence[str]] = None) -> torch.Tensor:
    """Flatten selected keys from one update dict into a single vector."""
    if keys is None:
        keys = [k for k, v in update.items() if _is_usable_tensor(k, v)]
    vec = []
    for key in keys:
        vec.append(update[key].detach().float().reshape(-1))
    if not vec:
        return torch.empty(0)
    return torch.cat(vec, dim=0)


def vector_to_update(vec: torch.Tensor, reference_update: StateDict, keys: Sequence[str]) -> StateDict:
    """Reshape a flat vector back to an update dict using a reference update dict."""
    out = {}
    pointer = 0
    for key in keys:
        ref = reference_update[key]
        numel = ref.numel()
        out[key] = vec[pointer:pointer + numel].reshape_as(ref).to(device=ref.device, dtype=ref.dtype)
        pointer += numel
    return out


def stack_updates(updates: Sequence[StateDict], keys: Optional[Sequence[str]] = None) -> Tuple[torch.Tensor, List[str]]:
    """Stack updates as an n x d matrix."""
    if not updates:
        return torch.empty(0), []
    if keys is None:
        keys = get_update_keys(updates)
    vectors = [flatten_update(update, keys) for update in updates]
    if not vectors or vectors[0].numel() == 0:
        return torch.empty((len(updates), 0)), list(keys)
    
    target_device = vectors[0].device
    aligned_vectors = [
        v.detach().to(device=target_device, dtype=torch.float32) 
        for v in vectors
    ]
    return torch.stack(aligned_vectors, dim=0), list(keys)

def filter_finite_updates(updates: Sequence[StateDict], keys: Optional[Sequence[str]] = None) -> List[StateDict]:
    """Remove updates containing NaN or Inf in the aggregated vector."""
    filtered = []
    for update in updates:
        vec = flatten_update(update, keys)
        if vec.numel() == 0 or torch.isfinite(vec).all():
            filtered.append(update)
    return filtered


def mean_update(updates: Sequence[StateDict], keys: Optional[Sequence[str]] = None) -> StateDict:
    """Coordinate-wise mean of a list of update dicts."""
    if not updates:
        return {}
    if keys is None:
        keys = get_update_keys(updates)
    aggregate = {}
    for key in keys:
        target_device = updates[0][key].device
        target_dtype = updates[0][key].dtype
        
        aggregate[key] = torch.mean(
            torch.stack(
                [upd[key].detach().to(device=target_device, dtype=torch.float32) for upd in updates], 
                dim=0
            ), 
            dim=0
        ).to(device=target_device, dtype=target_dtype)
    return aggregate

def add_update_to_model(global_model, aggregate_update: StateDict):
    """Add an aggregated delta/update to a model and return the updated model."""
    if not aggregate_update:
        return global_model
    new_state = copy.deepcopy(global_model.state_dict())
    for key, delta in aggregate_update.items():
        if key in new_state and torch.is_tensor(delta) and new_state[key].shape == delta.shape:
            if torch.is_floating_point(new_state[key]):
                new_state[key] = new_state[key] + delta.to(device=new_state[key].device, dtype=new_state[key].dtype)
    global_model.load_state_dict(new_state, strict=True)
    return global_model


def robust_mad_scores(values: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Median/MAD robust z-like scores for a 1-D tensor."""
    values = values.float()
    median = torch.median(values)
    mad = torch.median(torch.abs(values - median))
    return torch.abs(values - median) / (1.4826 * mad + eps)


def topk_sparsify_update(update: StateDict, keep_ratio: float = 0.3, keys: Optional[Sequence[str]] = None) -> StateDict:
    """Return a new update where only top-k magnitude coordinates are kept per tensor."""
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    if keys is None:
        keys = [k for k, v in update.items() if _is_usable_tensor(k, v)]
    out = copy.deepcopy(update)
    for key in keys:
        tensor = update[key]
        if tensor.numel() == 0:
            continue
        flat_abs = tensor.detach().abs().reshape(-1)
        k = max(1, int(round(keep_ratio * flat_abs.numel())))
        if k >= flat_abs.numel():
            continue
        threshold = torch.topk(flat_abs, k, largest=True).values[-1]
        mask = (tensor.detach().abs() >= threshold).to(dtype=tensor.dtype, device=tensor.device)
        out[key] = tensor * mask
    return out
