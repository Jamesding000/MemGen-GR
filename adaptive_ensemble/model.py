from typing import List, Tuple, Dict, Iterable
import math
import numpy as np


def min_max_scale(scores: Iterable[float]) -> np.ndarray:
    scores = np.asarray(list(scores), dtype=float)
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi <= lo:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def blend_predictions(sas_items: List[int], sas_scores: List[float],
                      tiger_items: List[int], tiger_scores: List[float],
                      alpha: float, ensemble_method: str,
                      normalization: str) -> List[int]:
    """Blend two ranked prediction lists into a single score-based ranking.

    Scores from each model are min–max normalised and combined as
    alpha * sasrec + (1 - alpha) * tiger. The ``ensemble_method`` and
    ``normalization`` arguments are accepted for API compatibility and
    are ignored.
    """
    s_norm = min_max_scale(sas_scores)
    t_norm = min_max_scale(tiger_scores)
    score_map: Dict[int, Dict[str, float]] = {}
    for it, sc in zip(sas_items, s_norm):
        score_map[it] = {"s": float(sc), "t": 0.0}
    for it, sc in zip(tiger_items, t_norm):
        if it not in score_map:
            score_map[it] = {"s": 0.0, "t": float(sc)}
        else:
            score_map[it]["t"] = float(sc)
    fused = [(it, alpha * scs["s"] + (1.0 - alpha) * scs["t"])
             for it, scs in score_map.items()]
    fused.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in fused]


class FixedEnsemble:
    """Ensemble that blends scores with a constant mixing weight alpha."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def blend(self, sas_items: List[int], sas_scores: List[float],
              tiger_items: List[int], tiger_scores: List[float],
              msp: float = None) -> Tuple[List[int], float]:
        ranked = blend_predictions(
            sas_items, sas_scores, tiger_items, tiger_scores,
            self.alpha, "score", "min_max",
        )
        return ranked, self.alpha


class AdaptiveEnsemble:
    """Adaptive ensemble that uses MSP to compute a sigmoid blending weight."""

    def __init__(self, k_steepness: float = 10.0, tau_threshold: float = 0.5):
        self.k_steepness = k_steepness
        self.tau_threshold = tau_threshold

    def compute_alpha(self, msp: float) -> float:
        """Compute the blending weight alpha from an MSP value."""
        k = self.k_steepness
        tau = self.tau_threshold
        # sigmoid weighting function
        return 1.0 / (1.0 + math.exp(-k * (msp - tau)))

    def blend(self, sas_items: List[int], sas_scores: List[float],
              tiger_items: List[int], tiger_scores: List[float],
              msp: float = None) -> Tuple[List[int], float]:
        alpha = self.compute_alpha(msp if msp is not None else 0.5)
        ranked = blend_predictions(
            sas_items, sas_scores, tiger_items, tiger_scores,
            alpha, "score", "min_max",
        )
        return ranked, alpha
