from typing import List, Tuple

from adaptive_ensemble.grid_search import blend_predictions


class FixedEnsemble:
    """Fixed-weight ensemble with a constant blending coefficient alpha."""

    def __init__(self, alpha: float, ensemble_method: str = "score",
                 normalization: str = "min_max"):
        self.alpha = alpha
        self.ensemble_method = ensemble_method
        self.normalization = normalization

    def blend(self, sas_items: List[int], sas_scores: List[float],
              tiger_items: List[int], tiger_scores: List[float],
              msp: float = None) -> Tuple[List[int], float]:
        ranked = blend_predictions(
            sas_items, sas_scores, tiger_items, tiger_scores,
            self.alpha, self.ensemble_method, self.normalization,
        )
        return ranked, self.alpha
