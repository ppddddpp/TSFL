import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.special import softmax

class CustomIsolationForest:
    def __init__(self, n_estimators=100, random_state=2709, n_jobs=-1, n_ensembles=5, 
                    max_samples=0.25, max_features=0.5, contamination="auto"):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.n_ensembles = n_ensembles
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination

    def _evaluate_sharpen_objective(self, weights: np.ndarray, gamma: float, eps: float = 1e-15) -> dict:
        """
        Evaluate the smooth objective:
            J(gamma) = 4*c*(1-c) * H(gamma)/H(1) * (1 - p_top(gamma)) / (1 - 1/n)

        where:
            p_i(gamma) = w_i^gamma / sum_j w_j^gamma
        """
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")

        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, eps, None)
        w = w / np.sum(w)

        n = len(w)
        if n < 2:
            raise ValueError("Need at least 2 weights.")

        a = np.log(w)                       # a_i = log(w_i)
        max_idx = int(np.argmax(w))        # assume unique top is fine in normal use
        a_max = float(a[max_idx])

        # Baseline statistics at gamma = 1
        mu_1 = float(np.sum(w * a))
        H_1 = float(-mu_1)
        den_c = a_max - mu_1
        den_D = 1.0 - 1.0 / n

        if den_c <= eps:
            # distribution too degenerate to sharpen meaningfully
            return {
                "gamma": float(gamma),
                "J": 0.0,
                "J_prime": 0.0,
                "p": w.copy(),
                "mu": mu_1,
                "v": 0.0,
                "c": 0.0,
                "B": 0.0,
                "H": H_1,
                "S": 1.0,
                "p_top": float(w[max_idx]),
                "D": (1.0 - float(w[max_idx])) / max(den_D, eps),
            }

        # p_i(gamma) = exp(gamma * log(w_i)) / Z
        z = np.exp(gamma * a)
        Z = float(np.sum(z))
        p = z / Z

        # mu(gamma) and variance under p(gamma)
        mu = float(np.sum(p * a))
        v = float(np.sum(p * (a - mu) ** 2))

        # c(gamma) in [0,1] theoretically
        c = (mu - mu_1) / den_c

        # Moderate-contrast term
        B = 4.0 * c * (1.0 - c)

        # Entropy-retention term
        H = float(-np.sum(p * np.log(np.clip(p, eps, None))))
        S = H / max(H_1, eps)

        # Anti-domination term
        p_top = float(p[max_idx])
        D = (1.0 - p_top) / max(den_D, eps)

        # Composite objective
        J = B * S * D

        # ===== derivatives =====
        # mu'(gamma) = Var_p(a) = v
        c_prime = v / den_c
        B_prime = 4.0 * c_prime * (1.0 - 2.0 * c)

        # H'(gamma) = -gamma * v
        S_prime = -(gamma * v) / max(H_1, eps)

        # p_top'(gamma) = p_top * (a_max - mu)
        p_top_prime = p_top * (a_max - mu)
        D_prime = -p_top_prime / max(den_D, eps)

        # J'(gamma) = B'SD + BS'D + BSD'
        J_prime = B_prime * S * D + B * S_prime * D + B * S * D_prime

        return {
            "gamma": float(gamma),
            "J": float(J),
            "J_prime": float(J_prime),
            "p": p,
            "mu": float(mu),
            "v": float(v),
            "c": float(c),
            "B": float(B),
            "H": float(H),
            "S": float(S),
            "p_top": float(p_top),
            "D": float(D),
        }

    def _find_optimal_sharpen_gamma(
        self,
        raw_weights: np.ndarray,
        gamma_min: float = 1.0,
        gamma_step: float = 0.25,
        gamma_max: float = 10.0,
        tol: float = 1e-10,
        max_iter: int = 200
    ) -> tuple[float, dict]:
        """
        Find gamma* by solving J'(gamma)=0 with:
        1) bracket search
        2) bisection on the derivative
        """
        left = self._evaluate_sharpen_objective(raw_weights, gamma_min)

        # If derivative is already non-positive at gamma_min,
        # the best point is near the left boundary.
        if left["J_prime"] <= 0:
            return float(gamma_min), left

        g_lo = gamma_min
        g_hi = gamma_min + gamma_step

        # Step 1: bracket a root of J'(gamma)
        while g_hi <= gamma_max:
            right = self._evaluate_sharpen_objective(raw_weights, g_hi)
            if right["J_prime"] <= 0:
                break
            g_lo = g_hi
            g_hi += gamma_step
        else:
            # No derivative sign change found up to gamma_max.
            # Fallback: compare objective at both ends.
            left_eval = self._evaluate_sharpen_objective(raw_weights, gamma_min)
            right_eval = self._evaluate_sharpen_objective(raw_weights, gamma_max)
            best = left_eval if left_eval["J"] >= right_eval["J"] else right_eval
            return float(best["gamma"]), best

        f_lo = self._evaluate_sharpen_objective(raw_weights, g_lo)["J_prime"]
        f_hi = self._evaluate_sharpen_objective(raw_weights, g_hi)["J_prime"]

        # Safety fallback if no sign change
        if f_lo * f_hi > 0:
            left_eval = self._evaluate_sharpen_objective(raw_weights, g_lo)
            right_eval = self._evaluate_sharpen_objective(raw_weights, g_hi)
            best = left_eval if left_eval["J"] >= right_eval["J"] else right_eval
            return float(best["gamma"]), best

        # Step 2: bisection on J'(gamma)=0
        for _ in range(max_iter):
            g_mid = 0.5 * (g_lo + g_hi)
            mid = self._evaluate_sharpen_objective(raw_weights, g_mid)
            f_mid = mid["J_prime"]

            if abs(f_mid) < tol or abs(g_hi - g_lo) < tol:
                return float(g_mid), mid

            if f_lo * f_mid > 0:
                g_lo = g_mid
                f_lo = f_mid
            else:
                g_hi = g_mid
                f_hi = f_mid

        g_mid = 0.5 * (g_lo + g_hi)
        mid = self._evaluate_sharpen_objective(raw_weights, g_mid)
        return float(g_mid), mid

    def compute_aggregation_weights(
            self, 
            feature_matrix: dict, 
            lower_bound_weight: float=1e-5,
            clip_max: float = 10,
            beta_min: float = 2.0,
            beta_max: float = 6.0,
            alpha_sensitivity: float = 0.5,
            gamma_0: float = 0.1,
            gamma_min: float = 1.0,
            gamma_step: float = 0.25,
            gamma_max: float = 10.0,
            gamma_tol: float = 1e-10,
            gamma_max_iter: int = 200,
            mode: str = "full_tsfl",
            sharpen: bool = True
        ) -> tuple:
        client_ids = list(feature_matrix.keys())
        feature_names = list(feature_matrix[client_ids[0]].keys())

        X_real = []
        # Convert dict to 2D NumPy array [n_clients, n_features]
        for cid in client_ids:
            client_row = []  # Create a new row for this specific client
            for f in feature_names:
                client_row.append(feature_matrix[cid][f])
            X_real.append(client_row) # Append the complete row to the matrix

        X_real = np.array(X_real, dtype=np.float64)
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_real)

        all_forest_scores = []
        for i in range(self.n_ensembles):
            model = IsolationForest(
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,   # Sub-sampling to break collusion
                max_features=self.max_features,   # Feature bagging to force diversity
                contamination=self.contamination,
                random_state=self.random_state + i, # Shift the random state for each forest
                n_jobs=self.n_jobs
            )
            
            model.fit(X_scaled)

            # Sklearn using higher is better (less anomalous)
            # But we need lower is better (higher energy = more anomalous)
            # So we negate the scores
            scores = -model.score_samples(X_scaled)
            all_forest_scores.append(scores)
        
        # Aggregate anomaly scores
        raw_anomaly_scores = np.mean(all_forest_scores, axis=0)

        if mode == "baseline":
            # Baseline 0: No defense. Equal weights for everyone.
            # Proves that Federated Learning inherently fails under attack without an anomaly scoring system.
            n_clients = len(client_ids)
            soft_weights = np.ones(n_clients) / n_clients
            logged_energies = np.zeros(n_clients)
            raw_weights = soft_weights.copy()
        
        elif mode == "softmax":
            # Temperature-scaled softmax aggregation
            min_score = np.min(raw_anomaly_scores)
            true_energies = raw_anomaly_scores - min_score
            
            # Softmax aggregation
            # Calculate adaptive temperature (tau) based on energy dispersion
            energy_mad = max(np.median(np.abs(true_energies - np.median(true_energies))), 1e-6) # MAD = mean(|x - median(x)|)
            tau = np.clip(energy_mad * 1.4826, 0.1, 1.0) # 1.4826 is the Fisher constant standard deviation
            
            # Convert anomaly energies to aggregation weights using Temperature-scaled Softmax.
            scaled_energies = -true_energies / tau
            soft_weights = softmax(scaled_energies)

            sum_weights = np.sum(soft_weights)
            if sum_weights > 0:
                soft_weights = soft_weights / sum_weights
            else:
                soft_weights = np.ones_like(soft_weights) / len(soft_weights)
            logged_energies = raw_anomaly_scores
            raw_weights = soft_weights.copy()

        elif mode == 'full_tsfl':
            min_score = np.min(raw_anomaly_scores)
            true_energies = raw_anomaly_scores - min_score
            
            # Calculate sigma using MAD for robust scaling
            median_e = float(np.median(true_energies))
            mad_e = float(np.median(np.abs(true_energies - median_e)))
            safe_mad_e = max(mad_e, 1e-6)

            robust_std = safe_mad_e * 1.4826 # Fisher consistent scaling factor for normal distribution
            sigma = 3.0 * robust_std        
            min_score = np.min(raw_anomaly_scores)
            true_energies = raw_anomaly_scores - min_score

            # IQR
            p90, p75, p50, p25, p10 = np.percentile(true_energies, [90, 75, 50, 25, 10])
            iqr = max(p75 - p25, 1e-6)

            # Quantile-based robust kurtosis (tail index)
            robust_kurtosis = (p90 - p10) / iqr
            kappa_shifted = max(0.0, robust_kurtosis - 2.0)

            # Bowley's skewness (robust asymmetry measure)
            robust_skewness = ((p75 - p50) - (p50 - p25)) / iqr

            # Robust bimodality coefficient (BC)
            robust_bc = (robust_skewness**2 + 1) / max(robust_kurtosis, 1e-6)
            if robust_bc > 0.555 and robust_kurtosis < 3.0:
                kappa_shifted = 0.0

            # Adaptive beta scaling based on kurtosis to control sensitivity
            beta = beta_min + (beta_max - beta_min) * np.tanh(alpha_sensitivity * kappa_shifted)
            
            # Lambda stretch factor based on the quantile-based robust kurtosis to further separate outliers in heavy-tailed distributions
            energy_gamma = gamma_0 * np.log1p(beta)

            scaled_energies = np.clip(true_energies / sigma, 0.0, clip_max)

            # Base soft weights from the energy model
            raw_weights = np.exp(- (scaled_energies ** beta) - energy_gamma * scaled_energies)
            raw_weights = np.maximum(raw_weights, lower_bound_weight)

            # Find optimal sharpening gamma to further enhance contrast between honest and anomalous clients
            if sharpen:
                sharpen_gamma, gamma_stats = self._find_optimal_sharpen_gamma(
                    raw_weights=raw_weights,
                    gamma_min=gamma_min,
                    gamma_step=gamma_step,
                    gamma_max=gamma_max,
                    tol=gamma_tol,
                    max_iter=gamma_max_iter
                )
                sharpened_weights = np.power(raw_weights, sharpen_gamma)
                self.last_sharpen_gamma = float(sharpen_gamma)
                self.last_sharpen_objective = gamma_stats
            else:
                sharpened_weights = raw_weights

            # L1 Normalization
            sum_weights = np.sum(sharpened_weights)
            if sum_weights > 0:
                soft_weights = sharpened_weights / sum_weights
            else:
                soft_weights = np.ones_like(sharpened_weights) / len(sharpened_weights)
            
            logged_energies = raw_anomaly_scores

        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Format the output dictionaries
        final_weights = dict(zip(client_ids, soft_weights.astype(float)))
        raw_energy_log = dict(zip(client_ids, logged_energies.astype(float)))
        soft_weights_log = dict(zip(client_ids, raw_weights.astype(float)))
        
        return final_weights, raw_energy_log, soft_weights_log