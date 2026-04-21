import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List, Optional
import scipy.stats

from .step_info import StepInfo
from .base_metric_hook import BaseMetricHook

class ObsHook(BaseMetricHook):
    """
    Extracts observation-based metafeatures for Instance Space Analysis.
    Captures variance, temporal volatility, sparsity, and intrinsic 
    dimensionality via PCA.
    """

    def __init__(self):
        self._reset_stats()

    def _reset_stats(self) -> None:
        # We store flattened observations to handle both 1D and 2D kinematics arrays
        self.all_observations: List[np.ndarray] = []
        self.step_differences: List[float] = []
        
        # Track the last observation to compute temporal volatility
        self.last_obs: Optional[np.ndarray] = None

    def on_probe_start(self) -> None:
        self._reset_stats()

    def on_episode_start(self) -> None:
        self.last_obs = None

    def on_step(self, context: StepInfo) -> None:
        # Flatten the observation (highway-env is usually V x F matrix)
        flat_obs = np.array(context.observation).flatten()
        self.all_observations.append(flat_obs)

        # Calculate volatility: L2 norm of the difference between current and last observation
        if self.last_obs is not None:
            diff = np.linalg.norm(flat_obs - self.last_obs)
            self.step_differences.append(float(diff))
            
        self.last_obs = flat_obs

    def on_episode_end(self) -> None:
        # We don't necessarily need to do anything here unless you want 
        # to calculate PCA per-episode rather than across the whole probe.
        # For ISA, aggregating the whole probe's state space is usually better.
        pass

    def finalize(self) -> Dict[str, float]:
        # Convert to an N x D matrix where N is total steps, D is flattened features
        X = np.vstack(self.all_observations)
        
        # 1. Global Distribution & Sparsity
        obs_mean = float(np.mean(X))
        obs_std = float(np.std(X))
        # Sparsity: proportion of elements that are exactly 0 (or very close)
        sparsity = float(np.mean(np.isclose(X, 0.0, atol=1e-5)))

        # 2. Temporal Volatility Stats
        if self.step_differences:
            volatility_mean = float(np.mean(self.step_differences))
            volatility_max = float(np.max(self.step_differences))
        else:
            volatility_mean, volatility_max = 0.0, 0.0

        # 3. PCA / Intrinsic Dimensionality
        # We use min(n_samples, n_features) as the max possible components
        n_components = min(X.shape[0], X.shape[1])
        
        pca_dim_90 = 0
        pca_dim_95 = 0
        pca_var_1 = 0.0

        if n_components > 1:
            try:
                pca = PCA(n_components=n_components)
                pca.fit(X)
                
                # Explained variance of the most dominant feature (PC1)
                pca_var_1 = float(pca.explained_variance_ratio_[0])
                
                # Cumulative variance to find intrinsic dimensionality
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                
                # How many components to explain 90% and 95% of the variance?
                # (np.argmax returns the first index where the condition is true)
                pca_dim_90 = int(np.argmax(cumulative_variance >= 0.90) + 1)
                pca_dim_95 = int(np.argmax(cumulative_variance >= 0.95) + 1)
                
                # Edge case fallback if it never reaches the threshold (rare, but safe)
                if cumulative_variance[-1] < 0.90:
                    pca_dim_90 = n_components
                if cumulative_variance[-1] < 0.95:
                    pca_dim_95 = n_components

            except Exception:
                # Fallback if PCA fails (e.g., zero variance across all features)
                pass

        # 4. Attribute Entropy
        entropies = []
        for i in range(X.shape[1]):
            col = X[:, i]
            if np.std(col) < 1e-5:
                entropies.append(0.0)
            else:
                counts, _ = np.histogram(col, bins='auto')
                prob = counts / np.sum(counts)
                entropies.append(float(scipy.stats.entropy(prob)))
        
        obs_entropy_mean = float(np.mean(entropies)) if entropies else 0.0
        obs_entropy_sum = float(np.sum(entropies)) if entropies else 0.0

        return {
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "obs_sparsity": sparsity,
            "obs_volatility_mean": volatility_mean,
            "obs_volatility_max": volatility_max,
            "pca_explained_var_pc1": pca_var_1,
            "pca_components_90_var": float(pca_dim_90),
            "pca_components_95_var": float(pca_dim_95),
            "obs_entropy_mean": obs_entropy_mean,
            "obs_entropy_sum": obs_entropy_sum,
        }
