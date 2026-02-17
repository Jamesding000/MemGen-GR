from typing import List, Tuple

from adaptive_ensemble.grid_search import (
    blend_predictions, sigmoid_msp, linear_msp,
)


class AdaptiveEnsemble:
    """MSP-adaptive ensemble where alpha varies per sample based on confidence."""

    def __init__(self, ensemble_method: str = "score", normalization: str = "min_max",
                 weighting_fn: str = "sigmoid", k_steepness: float = 10.0,
                 tau_threshold: float = 0.5, w: float = 1.0, b: float = 0.0):
        self.ensemble_method = ensemble_method
        self.normalization = normalization
        self.weighting_fn = weighting_fn
        self.k_steepness = k_steepness
        self.tau_threshold = tau_threshold
        self.w = w
        self.b = b

    def compute_alpha(self, msp: float) -> float:
        if self.weighting_fn == "linear":
            return linear_msp(msp, self.w, self.b)
        return sigmoid_msp(msp, self.k_steepness, self.tau_threshold)

    def blend(self, sas_items: List[int], sas_scores: List[float],
              tiger_items: List[int], tiger_scores: List[float],
              msp: float = None) -> Tuple[List[int], float]:
        alpha = self.compute_alpha(msp if msp is not None else 0.5)
        ranked = blend_predictions(
            sas_items, sas_scores, tiger_items, tiger_scores,
            alpha, self.ensemble_method, self.normalization,
        )
        return ranked, alpha
