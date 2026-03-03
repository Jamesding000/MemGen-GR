"""
Section 5.3: Performance Analysis for Adaptive Ensemble
Perform a grid search over the hyperparameters of the adaptive ensemble
and report its performance compared to the single models and the fixed ensemble.
"""

import argparse
import math
import os
from typing import Dict, List, Tuple, Iterable
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from adaptive_ensemble.model import FixedEnsemble, AdaptiveEnsemble


def compute_ndcg_at_k(items: List[int], target: int, k: int) -> float:
    top = items[:k]
    if target not in top:
        return 0.0
    rank = top.index(target)
    return 1.0 / math.log2(rank + 2.0)


def compute_recall_at_k(items: List[int], target: int, k: int) -> float:
    return 1.0 if target in items[:k] else 0.0


def _pct_improve(new: float, base: float) -> float:
    return (new - base) / base * 100.0 if base > 0 else 0.0


def _range_to_arange(r: List[float]) -> Tuple[float, float, float]:
    """Convert [min, max, step] to np.arange-compatible args."""
    return r[0], r[1] + 0.5 * r[2], r[2]


def load_sasrec_predictions(csv_path: str, split: str, n_predictions: int) -> Dict[int, Dict]:
    df = pd.read_csv(csv_path)
    if "top_items" in df.columns and "top_scores" in df.columns:
        items_col, scores_col = "top_items", "top_scores"
    elif "top10_items" in df.columns and "top10_scores" in df.columns:
        items_col, scores_col = "top10_items", "top10_scores"
    else:
        raise ValueError("SASRec CSV must have (top_items, top_scores) or (top10_items, top10_scores)")

    data: Dict[int, Dict] = {}
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"SASRec - {split}"):
        sample_id = int(getattr(row, "sample_idx"))
        data[sample_id] = {
            "items": list(eval(getattr(row, items_col)))[:n_predictions],
            "scores": list(eval(getattr(row, scores_col)))[:n_predictions],
            "target": int(getattr(row, "target_item")),
            "msp": float(getattr(row, "confidence_msp")),
        }
    return data


def load_tiger_predictions(csv_path: str, split: str, n_predictions: int) -> Dict[int, Dict]:
    df = pd.read_csv(csv_path)
    df["sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce")
    df["beam_rank"] = pd.to_numeric(df["beam_rank"], errors="coerce")
    df = df.dropna(subset=["sample_id", "beam_rank"])

    data: Dict[int, Dict] = {}
    for sample_id, group in tqdm(df.groupby("sample_id"), desc=f"TIGER - {split}"):
        top = group.sort_values("beam_rank").head(n_predictions)
        data[int(sample_id)] = {
            "items": top["pred_item"].astype(int).tolist(),
            "scores": top["beam_score"].astype(float).tolist(),
            "target": int(top["target_item"].iloc[0])
        }
    return data


def compute_single_model_metrics(sas_data: Dict[int, Dict], 
                                 tiger_data: Dict[int, Dict], 
                                 top_k: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    common_ids = sorted(set(sas_data) & set(tiger_data))
    sas_res, tig_res = {"ndcg": [], "recall": []}, {"ndcg": [], "recall": []}
    
    for sid in common_ids:
        s_rec, t_rec = sas_data[sid], tiger_data[sid]
        if s_rec["target"] != t_rec["target"]:
            continue
        
        target = s_rec["target"]
        sas_res["ndcg"].append(compute_ndcg_at_k(s_rec["items"], target, top_k))
        sas_res["recall"].append(compute_recall_at_k(s_rec["items"], target, top_k))
        tig_res["ndcg"].append(compute_ndcg_at_k(t_rec["items"], target, top_k))
        tig_res["recall"].append(compute_recall_at_k(t_rec["items"], target, top_k))

    return (
        {"ndcg": float(np.mean(sas_res["ndcg"])), "recall": float(np.mean(sas_res["recall"]))},
        {"ndcg": float(np.mean(tig_res["ndcg"])), "recall": float(np.mean(tig_res["recall"]))}
    )


def evaluate_ensemble(sas_data: Dict[int, Dict], tiger_data: Dict[int, Dict],
                      top_k: int, ensemble) -> Dict[str, float]:
    common_ids = sorted(set(sas_data) & set(tiger_data))
    ndcg_list, recall_list = [], []

    for sid in common_ids:
        s_rec, t_rec = sas_data[sid], tiger_data[sid]
        if s_rec["target"] != t_rec["target"]:
            continue
        
        target = s_rec["target"]
        ranked_items, _ = ensemble.blend(
            s_rec["items"], s_rec["scores"],
            t_rec["items"], t_rec["scores"],
            msp=float(s_rec["msp"])
        )

        ndcg_list.append(compute_ndcg_at_k(ranked_items, target, top_k))
        recall_list.append(compute_recall_at_k(ranked_items, target, top_k))

    if not ndcg_list:
        return {"ndcg": 0.0, "recall": 0.0}

    return {"ndcg": float(np.mean(ndcg_list)), "recall": float(np.mean(recall_list))}


def grid_search_fixed(sas_val: Dict[int, Dict], tiger_val: Dict[int, Dict],
                      alphas: Iterable[float], top_k: int) -> float:
    best_ndcg, best_alpha = -1.0, 0.5
    for a in tqdm(alphas, desc="Grid Search Fixed Ensemble"):
        res = evaluate_ensemble(sas_val, tiger_val, top_k, FixedEnsemble(float(a)))
        if res["ndcg"] > best_ndcg:
            best_ndcg, best_alpha = res["ndcg"], float(a)
    return best_alpha


def grid_search_adaptive(sas_val: Dict[int, Dict], tiger_val: Dict[int, Dict],
                        k_values: Iterable[float], tau_values: Iterable[float],
                        top_k: int) -> Tuple[float, float]:
    best_ndcg, best_params = -1.0, (0.0, 0.0)

    grid = list(itertools.product(k_values, tau_values))
    for k, tau in tqdm(grid, desc="Grid Search Dynamic Ensemble"):
        res = evaluate_ensemble(sas_val, tiger_val, top_k, AdaptiveEnsemble(float(k), float(tau)))
        if res["ndcg"] > best_ndcg:
            best_ndcg, best_params = res["ndcg"], (float(k), float(tau))
    return best_params


def main() -> None:
    p = argparse.ArgumentParser(description="Run Ensemble Experiment for a Single Dataset")
    p.add_argument("--dataset_name", required=True, help="Base dataset name (e.g., AmazonReviews2014)")
    p.add_argument("--category", default="", help="Dataset category (e.g., Beauty)")
    p.add_argument("--version", default="", help="Dataset version (e.g., Yelp_2020)")
    p.add_argument("--base_dir", default="outputs", help="Directory containing the prediction CSVs")
    p.add_argument("--top_k", type=int, default=10, help="K for NDCG and Recall")
    p.add_argument("--n_predictions", type=int, default=50, help="Top N candidates to consider per model")
    p.add_argument("--k_range", nargs=3, type=float, default=[1.0, 25.0, 4.0], help="Min, Max, Step for Steepness (k)")
    p.add_argument("--tau_range", nargs=3, type=float, default=[0.0, 0.5, 0.1], help="Min, Max, Step for Threshold (tau)")
    p.add_argument("--alpha_range", nargs=3, type=float, default=[0.0, 1.0, 0.1], help="Min, Max, Step for Fixed Alpha")
    args = p.parse_args()

    ds_id = args.dataset_name
    if args.version:
        ds_id = f"{args.dataset_name}-{args.version}"
    elif args.category:
        ds_id = f"{args.dataset_name}-{args.category}"

    sas_val_csv = os.path.join(args.base_dir, f"sasrec_predictions_with_scores_val_{ds_id}.csv")
    tig_val_csv = os.path.join(args.base_dir, f"tiger_predictions_with_scores_val_{ds_id}.csv")
    sas_tes_csv = os.path.join(args.base_dir, f"sasrec_predictions_with_scores_test_{ds_id}.csv")
    tig_tes_csv = os.path.join(args.base_dir, f"tiger_predictions_with_scores_test_{ds_id}.csv")

    # load inference results
    print(f"\nLoading data for {ds_id}...")
    sas_val = load_sasrec_predictions(sas_val_csv, "val", args.n_predictions)
    tig_val = load_tiger_predictions(tig_val_csv, "val", args.n_predictions)
    sas_test = load_sasrec_predictions(sas_tes_csv, "test", args.n_predictions)
    tig_test = load_tiger_predictions(tig_tes_csv, "test", args.n_predictions)

    # evaluate single models
    sas_test_metrics, tig_test_metrics = compute_single_model_metrics(sas_test, tig_test, args.top_k)
    best_single_ndcg = max(sas_test_metrics["ndcg"], tig_test_metrics["ndcg"])

    alphas = np.arange(*_range_to_arange(args.alpha_range))
    k_vals = np.arange(*_range_to_arange(args.k_range))
    tau_vals = np.arange(*_range_to_arange(args.tau_range))

    # evaluate fixed ensemble
    print(f"\nRunning Grid Search for Fixed Ensemble...")
    best_alpha = grid_search_fixed(sas_val, tig_val, alphas, args.top_k)
    fixed_test_res = evaluate_ensemble(sas_test, tig_test, args.top_k, FixedEnsemble(best_alpha))

    # evaluate adaptive ensemble
    print(f"Running Grid Search for Adaptive Ensemble...")
    best_k, best_tau = grid_search_adaptive(sas_val, tig_val, k_vals, tau_vals, args.top_k)
    adaptive_test_res = evaluate_ensemble(sas_test, tig_test, args.top_k, AdaptiveEnsemble(best_k, best_tau))

    # report results
    print(f"\n{'='*75}")
    print(f" EXPERIMENT REPORT: {ds_id} (Top-{args.top_k})")
    print(f"{'='*75}")
    
    print(f"[Single Models]")
    print(f"  SASRec        | NDCG: {sas_test_metrics['ndcg']:.4f} | Recall: {sas_test_metrics['recall']:.4f}")
    print(f"  TIGER         | NDCG: {tig_test_metrics['ndcg']:.4f} | Recall: {tig_test_metrics['recall']:.4f}")
    print(f"  Best Baseline | NDCG: {best_single_ndcg:.4f}")
    print(f"{'-'*75}")
    
    print(f"[Fixed Ensemble]")
    print(f"  Best Params   : alpha = {best_alpha:.2f}")
    print(f"  Test Metrics  : NDCG: {fixed_test_res['ndcg']:.4f} | Recall: {fixed_test_res['recall']:.4f}")
    print(f"  vs Best Single: {_pct_improve(fixed_test_res['ndcg'], best_single_ndcg):+.2f}%")
    print(f"{'-'*75}")
    
    print(f"[Adaptive Ensemble]")
    print(f"  Best Params   : k = {best_k:.2f}, tau = {best_tau:.2f}")
    print(f"  Test Metrics  : NDCG: {adaptive_test_res['ndcg']:.4f} | Recall: {adaptive_test_res['recall']:.4f}")
    print(f"  vs Best Single: {_pct_improve(adaptive_test_res['ndcg'], best_single_ndcg):+.2f}%")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()