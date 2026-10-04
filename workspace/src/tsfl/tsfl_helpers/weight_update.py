import torch

class WeightUpdater:
    def __init__(
            self, 
            server_lr=1.0, 
            beta_momentum=0.9, 
            beta_variance=0.999, 
            tau=1e-3
        ):
        self.server_lr = server_lr

        # Hyperparameters for advanced server optimizers (FedAdam / FedAvgM)
        self.beta_momentum = beta_momentum
        self.beta_variance = beta_variance
        self.tau = tau

        # Memory (State) for momentum across rounds
        self.m_t = None
        self.v_t = None

    def apply_update_method(self, update_method, global_state, client_deltas, client_weights):
        if update_method == "fedavg":
            return self._fedavg(global_state, client_deltas, client_weights)
        elif update_method == "fedavgm":
            return self._fedavgm(global_state, client_deltas, client_weights)
        elif update_method == "fedadam":
            return self._fedadam(global_state, client_deltas, client_weights)
        else:
            raise ValueError(f"Unknown update method: {update_method}")

    @torch.no_grad()
    def _fedavg(self, global_state, client_deltas, client_weights):
        empty_global = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items() if v.is_floating_point()}
        
        # Sum up the weighted client deltas
        for cid, delta_data in client_deltas.items():
            w = client_weights[cid]
            client_tensors = delta_data.get("weights") or delta_data
            
            for k in empty_global.keys():
                empty_global[k] += client_tensors[k].to(device=global_state[k].device, dtype=torch.float32) * w
                
        # Add the aggregated delta back to the old global model
        new_global = {}
        for k, v in global_state.items():
            # If it's an integer counter, just copy it directly without math
            if not v.is_floating_point():
                new_global[k] = v.clone()
                continue
                
            update_amount = (empty_global[k] * self.server_lr).to(v.dtype)
            new_global[k] = v + update_amount
            
        return new_global
    @torch.no_grad()
    def _fedavgm(self, global_state, client_deltas, client_weights):
        pseudo_gradient = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items() if v.is_floating_point()}
        
        for cid, delta_data in client_deltas.items():
            w = client_weights[cid]
            client_tensors = delta_data.get("weights") or delta_data
            for k in pseudo_gradient.keys():
                pseudo_gradient[k] += client_tensors[k].to(device=global_state[k].device, dtype=torch.float32) * w

        # Initialize momentum memory dynamically
        if self.m_t is None:
            self.m_t = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items() if v.is_floating_point()}

        new_global = {}
        # Apply Server Momentum update
        for k, v in global_state.items():
            # If it's an integer counter, just copy it directly without math
            if not v.is_floating_point():
                new_global[k] = v.clone()
                continue

            self.m_t[k] = (self.beta_momentum * self.m_t[k]) + ((1 - self.beta_momentum) * pseudo_gradient[k])
            
            update_amount = (self.m_t[k] * self.server_lr).to(v.dtype)
            new_global[k] = v + update_amount
            
        return new_global
    
    @torch.no_grad()
    def _fedadam(self, global_state, client_deltas, client_weights):
        pseudo_gradient = {}
        for k, v in global_state.items():
            if v.is_floating_point():
                pseudo_gradient[k] = torch.zeros_like(v, dtype=torch.float32)

        # Accumulate gradients ONLY for the floating-point keys
        for cid, delta_data in client_deltas.items():
            w = client_weights[cid]
            client_tensors = delta_data.get("weights") or delta_data
            
            for k in pseudo_gradient.keys():
                pseudo_gradient[k] += client_tensors[k].to(device=global_state[k].device, dtype=torch.float32) * w

        # Initialize momentum and variance tracking dynamically
        if self.m_t is None:
            self.m_t = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items() if v.is_floating_point()}
            self.v_t = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items() if v.is_floating_point()}

        new_global = {}
        # Apply updates dynamically
        for k, v in global_state.items():
            # If it is an integer/counter (like num_batches_tracked), just copy it directly
            if not v.is_floating_point():
                new_global[k] = v.clone()
                continue
                
            # Adam math (Momentum + Variance tracking)
            self.m_t[k] = (self.beta_momentum * self.m_t[k]) + ((1 - self.beta_momentum) * pseudo_gradient[k])
            self.v_t[k] = (self.beta_variance * self.v_t[k]) + ((1 - self.beta_variance) * (pseudo_gradient[k] ** 2))
            
            # Apply adaptive update
            step = self.m_t[k] / (torch.sqrt(self.v_t[k]) + self.tau)
            
            # Cast the final step back to the original type
            update_amount = (step * self.server_lr).to(v.dtype)
            new_global[k] = v + update_amount
            
        return new_global