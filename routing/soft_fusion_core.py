import math
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def sigmoid_msp(x: float, k: float, tau: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - tau)))


def linear_msp(x: float, w: float, b: float) -> float:
    v = w * x + b
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def min_max_scale(scores: Iterable[float]) -> np.ndarray:
    scores = np.asarray(list(scores), dtype=float)
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi <= lo:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def z_score_normalize(scores: Iterable[float]) -> np.ndarray:
    scores = np.asarray(list(scores), dtype=float)
    if scores.size == 0:
        return scores
    mu, sigma = scores.mean(), scores.std()
    if sigma == 0.0:
        return np.zeros_like(scores)
    return (scores - mu) / sigma


def compute_ndcg_at_k(items: List[int], target: int, k: int) -> float:
    top = items[:k]
    if target not in top:
        return 0.0
    rank = top.index(target)
    return 1.0 / math.log2(rank + 2.0)


def compute_recall_at_k(items: List[int], target: int, k: int) -> float:
    return 1.0 if target in items[:k] else 0.0


def load_sasrec_predictions(csv_path: str, split: str, n_predictions: int) -> Dict[int, Dict]:
    df = pd.read_csv(csv_path)
    if "top_items" in df.columns and "top_scores" in df.columns:
        items_col, scores_col = "top_items", "top_scores"
    elif "top10_items" in df.columns and "top10_scores" in df.columns:
        items_col, scores_col = "top10_items", "top10_scores"
    else:
        raise ValueError("SASRec CSV must have (top_items, top_scores) or (top10_items, top10_scores)")

    data: Dict[int, Dict] = {}
    it = tqdm(df.itertuples(index=False), total=len(df), desc=f"SASRec - {split}")  # type: ignore[arg-type]
    for row in it:
        sample_id = int(getattr(row, "sample_idx"))
        items = list(eval(getattr(row, items_col)))[:n_predictions]
        scores = list(eval(getattr(row, scores_col)))[:n_predictions]
        data[sample_id] = {
            "user_id": int(getattr(row, "user_id")),
            "items": items,
            "scores": scores,
            "target": int(getattr(row, "target_item")),
            "msp": float(getattr(row, "confidence_msp")),
        }
    return data


def load_tiger_predictions(csv_path: str, split: str, n_predictions: int) -> Dict[int, Dict]:
    df = pd.read_csv(csv_path)
    df["sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce")
    df["beam_rank"] = pd.to_numeric(df["beam_rank"], errors="coerce")
    df = df.dropna(subset=["sample_id", "beam_rank"])
    df["sample_id"] = df["sample_id"].astype(int)
    df["beam_rank"] = df["beam_rank"].astype(int)

    data: Dict[int, Dict] = {}
    for sample_id, group in tqdm(df.groupby("sample_id"), desc=f"TIGER - {split}"):
        group = group.sort_values("beam_rank")
        top = group.head(n_predictions)
        items = top["pred_item"].astype(int).tolist()
        scores = top["beam_score"].astype(float).tolist()
        target = int(top["target_item"].iloc[0])
        data[int(sample_id)] = {"items": items, "scores": scores, "target": target}
    return data


def evaluate_fusion(
    sas_data: Dict[int, Dict],
    tiger_data: Dict[int, Dict],
    top_k: int,
    fusion_method: str,
    normalization: str,
    weighting_mode: str,
    static_alpha: Optional[float] = None,
    weighting_fn: Optional[str] = None,
    k_steepness: float = 10.0,
    tau_threshold: float = 0.5,
    w: float = 1.0,
    b: float = 0.0,
) -> Dict:
    common_ids = sorted(set(sas_data.keys()) & set(tiger_data.keys()))
    details: List[Dict] = []

    for sid in common_ids:
        s_rec, t_rec = sas_data[sid], tiger_data[sid]
        if s_rec["target"] != t_rec["target"]:
            continue
        target = s_rec["target"]
        msp = float(s_rec["msp"])

        if weighting_mode == "static":
            alpha = float(static_alpha if static_alpha is not None else 0.5)
        elif weighting_fn == "linear":
            alpha = linear_msp(msp, w, b)
        else:
            alpha = sigmoid_msp(msp, k_steepness, tau_threshold)

        sas_items, sas_scores = s_rec["items"], s_rec["scores"]
        tiger_items, tiger_scores = t_rec["items"], t_rec["scores"]

        if fusion_method == "rank":
            rank_sas = {item: i for i, item in enumerate(sas_items)}
            rank_tig = {item: i for i, item in enumerate(tiger_items)}
            all_items = set(sas_items) | set(tiger_items)
            fused = []
            for item in all_items:
                r_s = rank_sas.get(item)
                r_t = rank_tig.get(item)
                score = 0.0
                if r_s is not None:
                    score += alpha / math.log2(r_s + 2.0)
                if r_t is not None:
                    score += (1.0 - alpha) / math.log2(r_t + 2.0)
                fused.append((item, score))
            fused.sort(key=lambda x: x[1], reverse=True)
            ranked_items = [x[0] for x in fused]
        else:
            if normalization == "standardize":
                s_norm = z_score_normalize(sas_scores)
                t_norm = z_score_normalize(tiger_scores)
            else:
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
            fused = []
            for it, scs in score_map.items():
                fused_score = alpha * scs["s"] + (1.0 - alpha) * scs["t"]
                fused.append((it, fused_score))
            fused.sort(key=lambda x: x[1], reverse=True)
            ranked_items = [x[0] for x in fused]

        ndcg = compute_ndcg_at_k(ranked_items, target, top_k)
        recall = compute_recall_at_k(ranked_items, target, top_k)
        sas_ndcg = compute_ndcg_at_k(sas_items, target, top_k)
        tig_ndcg = compute_ndcg_at_k(tiger_items, target, top_k)

        details.append(
            {
                "ndcg": ndcg,
                "recall": recall,
                "alpha": alpha,
                "sas_ndcg": sas_ndcg,
                "tiger_ndcg": tig_ndcg,
                "msp": msp,
            }
        )

    if not details:
        return {"avg_ndcg": 0.0, "avg_recall": 0.0, "avg_alpha": 0.0, "details": []}

    avg_ndcg = float(np.mean([d["ndcg"] for d in details]))
    avg_recall = float(np.mean([d["recall"] for d in details]))
    avg_alpha = float(np.mean([d["alpha"] for d in details]))
    return {"avg_ndcg": avg_ndcg, "avg_recall": avg_recall, "avg_alpha": avg_alpha, "details": details}


def grid_search_static_alpha(
    sas_val: Dict[int, Dict],
    tiger_val: Dict[int, Dict],
    alphas: Iterable[float],
    fusion_method: str,
    normalization: str,
    top_k: int,
    verbose: bool = False,
) -> Tuple[List[Tuple[float, float]], float]:
    curve: List[Tuple[float, float]] = []
    best_ndcg = -1.0
    best_alpha = 0.5
    for a in alphas:
        res = evaluate_fusion(
            sas_val,
            tiger_val,
            top_k=top_k,
            fusion_method=fusion_method,
            normalization=normalization,
            weighting_mode="static",
            static_alpha=float(a),
        )
        if verbose:
            print(f"Static alpha: {a:.2f}, NDCG: {res['avg_ndcg']:.4f}")
        curve.append((float(a), res["avg_ndcg"]))
        if res["avg_ndcg"] > best_ndcg:
            best_ndcg = res["avg_ndcg"]
            best_alpha = float(a)
    return curve, best_alpha


def grid_search_dynamic(
    sas_val: Dict[int, Dict],
    tiger_val: Dict[int, Dict],
    weighting_fn: str,
    p1_values: Iterable[float],
    p2_values: Iterable[float],
    fusion_method: str,
    normalization: str,
    top_k: int,
    verbose: bool = False,
) -> Dict:
    records = []
    best_ndcg = -1.0
    best_params = (0.0, 0.0)

    for p1 in p1_values:
        for p2 in p2_values:
            if weighting_fn == "sigmoid":
                res = evaluate_fusion(
                    sas_val,
                    tiger_val,
                    top_k=top_k,
                    fusion_method=fusion_method,
                    normalization=normalization,
                    weighting_mode="dynamic",
                    weighting_fn="sigmoid",
                    k_steepness=float(p1),
                    tau_threshold=float(p2),
                )
                if verbose:
                    print(f"Sigmoid: k={p1:.2f}, tau={p2:.2f}, NDCG: {res['avg_ndcg']:.4f}")
                records.append(
                    {"k": p1, "tau": p2, "ndcg": res["avg_ndcg"], "recall": res["avg_recall"], "avg_alpha": res["avg_alpha"]}
                )
            else:
                res = evaluate_fusion(
                    sas_val,
                    tiger_val,
                    top_k=top_k,
                    fusion_method=fusion_method,
                    normalization=normalization,
                    weighting_mode="dynamic",
                    weighting_fn="linear",
                    w=float(p1),
                    b=float(p2),
                )
                if verbose:
                    print(f"Linear: w={p1}, b={p2}, NDCG: {res['avg_ndcg']}")
                records.append(
                    {"w": p1, "b": p2, "ndcg": res["avg_ndcg"], "recall": res["avg_recall"], "avg_alpha": res["avg_alpha"]}
                )
            if res["avg_ndcg"] > best_ndcg:
                best_ndcg = res["avg_ndcg"]
                best_params = (float(p1), float(p2))

    grid = pd.DataFrame.from_records(records)
    return {"best_params": best_params, "best_val_ndcg": best_ndcg, "grid_table": grid}


def compute_single_model_baselines(
    sas_data: Dict[int, Dict],
    tiger_data: Dict[int, Dict],
    top_k: int,
) -> Tuple[float, float]:
    common_ids = sorted(set(sas_data.keys()) & set(tiger_data.keys()))
    sas_vals, tig_vals = [], []
    for sid in common_ids:
        s_rec, t_rec = sas_data[sid], tiger_data[sid]
        if s_rec["target"] != t_rec["target"]:
            continue
        target = s_rec["target"]
        sas_vals.append(compute_ndcg_at_k(s_rec["items"], target, top_k))
        tig_vals.append(compute_ndcg_at_k(t_rec["items"], target, top_k))
    return (float(np.mean(sas_vals)) if sas_vals else 0.0, float(np.mean(tig_vals)) if tig_vals else 0.0)

