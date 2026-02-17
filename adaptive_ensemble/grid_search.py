import argparse
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

# Utility functions
def sigmoid_msp(x: float, k: float, tau: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - tau)))


def linear_msp(x: float, w: float, b: float) -> float:
    v = w * x + b
    return np.clip(v, 0.0, 1.0)


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


def blend_predictions(sas_items: List[int], sas_scores: List[float],
                      tiger_items: List[int], tiger_scores: List[float],
                      alpha: float, ensemble_method: str,
                      normalization: str) -> List[int]:
    """Blend two ranked prediction lists into a single blended ranking."""
    if ensemble_method == "rank":
        rank_sas = {item: i for i, item in enumerate(sas_items)}
        rank_tig = {item: i for i, item in enumerate(tiger_items)}
        fused = []
        for item in set(sas_items) | set(tiger_items):
            score = 0.0
            r_s = rank_sas.get(item)
            r_t = rank_tig.get(item)
            if r_s is not None:
                score += alpha / math.log2(r_s + 2.0)
            if r_t is not None:
                score += (1.0 - alpha) / math.log2(r_t + 2.0)
            fused.append((item, score))
    else:
        s_norm = (z_score_normalize(sas_scores) if normalization == "standardize"
                  else min_max_scale(sas_scores))
        t_norm = (z_score_normalize(tiger_scores) if normalization == "standardize"
                  else min_max_scale(tiger_scores))
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

# Data loading
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

# Generic evaluation
def evaluate_ensemble(sas_data: Dict[int, Dict], tiger_data: Dict[int, Dict],
                      top_k: int, ensemble) -> Dict:
    """Evaluate an ensemble on paired SASRec/TIGER predictions."""
    common_ids = sorted(set(sas_data) & set(tiger_data))
    details: List[Dict] = []

    for sid in common_ids:
        s_rec, t_rec = sas_data[sid], tiger_data[sid]
        if s_rec["target"] != t_rec["target"]:
            continue
        target = s_rec["target"]
        msp = float(s_rec["msp"])

        ranked_items, alpha = ensemble.blend(
            s_rec["items"], s_rec["scores"],
            t_rec["items"], t_rec["scores"],
            msp=msp,
        )

        details.append({
            "ndcg": compute_ndcg_at_k(ranked_items, target, top_k),
            "recall": compute_recall_at_k(ranked_items, target, top_k),
            "alpha": alpha,
            "sas_ndcg": compute_ndcg_at_k(s_rec["items"], target, top_k),
            "tiger_ndcg": compute_ndcg_at_k(t_rec["items"], target, top_k),
            "msp": msp,
        })

    if not details:
        return {"avg_ndcg": 0.0, "avg_recall": 0.0, "avg_alpha": 0.0, "details": []}

    return {
        "avg_ndcg": float(np.mean([d["ndcg"] for d in details])),
        "avg_recall": float(np.mean([d["recall"] for d in details])),
        "avg_alpha": float(np.mean([d["alpha"] for d in details])),
        "details": details,
    }

# Grid search
def grid_search_fixed(sas_val: Dict[int, Dict], tiger_val: Dict[int, Dict],
                             alphas: Iterable[float], ensemble_method: str,
                             normalization: str, top_k: int,
                             verbose: bool = False) -> Tuple[List[Tuple[float, float]], float]:
    from adaptive_ensemble.fixed_ensemble import FixedEnsemble
    curve: List[Tuple[float, float]] = []
    best_ndcg, best_alpha = -1.0, 0.5
    for a in alphas:
        res = evaluate_ensemble(sas_val, tiger_val, top_k,
                              FixedEnsemble(float(a), ensemble_method, normalization))
        if verbose:
            print(f"  alpha={a:.2f}  NDCG={res['avg_ndcg']:.4f}")
        curve.append((float(a), res["avg_ndcg"]))
        if res["avg_ndcg"] > best_ndcg:
            best_ndcg, best_alpha = res["avg_ndcg"], float(a)
    return curve, best_alpha


def grid_search_dynamic(sas_val: Dict[int, Dict], tiger_val: Dict[int, Dict],
                        weighting_fn: str, p1_values: Iterable[float],
                        p2_values: Iterable[float], ensemble_method: str,
                        normalization: str, top_k: int,
                        verbose: bool = False) -> Dict:
    from adaptive_ensemble.adaptive_ensemble import AdaptiveEnsemble
    records: List[Dict] = []
    best_ndcg, best_params = -1.0, (0.0, 0.0)

    for p1 in p1_values:
        for p2 in p2_values:
            if weighting_fn == "sigmoid":
                ensemble = AdaptiveEnsemble(
                    ensemble_method, normalization, weighting_fn="sigmoid",
                    k_steepness=float(p1), tau_threshold=float(p2),
                )
                record: Dict = {"k": p1, "tau": p2}
            else:
                ensemble = AdaptiveEnsemble(
                    ensemble_method, normalization, weighting_fn="linear",
                    w=float(p1), b=float(p2),
                )
                record = {"w": p1, "b": p2}

            res = evaluate_ensemble(sas_val, tiger_val, top_k, ensemble)
            if verbose:
                tag = (f"k={p1:.2f}, tau={p2:.2f}" if weighting_fn == "sigmoid"
                       else f"w={p1:.2f}, b={p2:.2f}")
                print(f"  {tag}  NDCG={res['avg_ndcg']:.4f}")

            record.update({"ndcg": res["avg_ndcg"], "recall": res["avg_recall"],
                           "avg_alpha": res["avg_alpha"]})
            records.append(record)
            if res["avg_ndcg"] > best_ndcg:
                best_ndcg, best_params = res["avg_ndcg"], (float(p1), float(p2))

    return {"best_params": best_params, "best_val_ndcg": best_ndcg,
            "grid_table": pd.DataFrame.from_records(records)}


def compute_single_model_baselines(sas_data: Dict[int, Dict], tiger_data: Dict[int, Dict],
                                   top_k: int) -> Tuple[float, float]:
    common_ids = sorted(set(sas_data) & set(tiger_data))
    sas_vals, tig_vals = [], []
    for sid in common_ids:
        s_rec, t_rec = sas_data[sid], tiger_data[sid]
        if s_rec["target"] != t_rec["target"]:
            continue
        target = s_rec["target"]
        sas_vals.append(compute_ndcg_at_k(s_rec["items"], target, top_k))
        tig_vals.append(compute_ndcg_at_k(t_rec["items"], target, top_k))
    return (float(np.mean(sas_vals)) if sas_vals else 0.0,
            float(np.mean(tig_vals)) if tig_vals else 0.0)

# Parallel master runner
def parse_specs(raw: List[str]) -> List[Tuple[str, str, str]]:
    specs: List[Tuple[str, str, str]] = []
    for s in raw:
        s = s.strip()
        if "-" in s:
            name, suffix = s.split("-", 1)
            specs.append((name, "", suffix) if "Yelp" in name else (name, suffix, ""))
        else:
            specs.append((s, "", ""))
    return specs


def run_one(python_path: str, dataset_name: str, category: str, version: str,
            base_dir: str, output_dir: str, top_k: int, n_predictions: int,
            normalization: str, k_range: List[float], tau_range: List[float],
            w_range: List[float], b_range: List[float],
            alpha_range: List[float]) -> str:
    ds_id = (f"{dataset_name}-{version}" if version
             else f"{dataset_name}-{category}" if category else dataset_name)
    sas_val = os.path.join(base_dir, f"sasrec_predictions_with_scores_val_{ds_id}.csv")
    tig_val = os.path.join(base_dir, f"tiger_predictions_with_scores_val_{ds_id}.csv")
    sas_tes = os.path.join(base_dir, f"sasrec_predictions_with_scores_test_{ds_id}.csv")
    tig_tes = os.path.join(base_dir, f"tiger_predictions_with_scores_test_{ds_id}.csv")

    cmd = [
        python_path, "-m", "adaptive_ensemble.visualization",
        "--dataset_name", dataset_name,
        "--category", category,
        "--version", version,
        "--sas_val_csv", sas_val,
        "--tiger_val_csv", tig_val,
        "--sas_test_csv", sas_tes,
        "--tiger_test_csv", tig_tes,
        "--output_dir", output_dir,
        "--top_k", str(top_k),
        "--n_predictions", str(n_predictions),
        "--normalization", normalization,
        "--k_range", *[str(x) for x in k_range],
        "--tau_range", *[str(x) for x in tau_range],
        "--w_range", *[str(x) for x in w_range],
        "--b_range", *[str(x) for x in b_range],
        "--alpha_range", *[str(x) for x in alpha_range],
    ]
    subprocess.run(cmd, check=True)
    return ds_id


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run adaptive ensemble grid search across datasets in parallel")
    p.add_argument("--python_path", default="python")
    p.add_argument("--base_dir", default="outputs")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_predictions", type=int, default=50)
    p.add_argument("--normalization", choices=["min_max", "standardize"], default="min_max")
    p.add_argument("--max_workers", type=int, default=2)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--k_range", nargs=3, type=float, default=[1.0, 25.0, 4.0])
    p.add_argument("--tau_range", nargs=3, type=float, default=[0.0, 0.5, 0.1])
    p.add_argument("--w_range", nargs=3, type=float, default=[0.0, 10.0, 2.0])
    p.add_argument("--b_range", nargs=3, type=float, default=[0.0, 1.0, 0.2])
    p.add_argument("--alpha_range", nargs=3, type=float, default=[0.0, 1.0, 0.1])
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    specs = parse_specs(args.datasets)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(
                run_one, args.python_path, name, cat, ver, args.base_dir,
                args.output_dir, args.top_k, args.n_predictions, args.normalization,
                args.k_range, args.tau_range, args.w_range, args.b_range, args.alpha_range,
            ): f"{name}-{ver}" if ver else f"{name}-{cat}" if cat else name
            for name, cat, ver in specs
        }
        for fut in as_completed(futures):
            ds = futures[fut]
            try:
                fut.result()
                print(f"[Done] {ds}")
            except Exception as e:
                print(f"[Error] {ds}: {e}")


if __name__ == "__main__":
    main()
