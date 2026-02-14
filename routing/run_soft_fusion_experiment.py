#!/usr/bin/env python3

import argparse
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
import os

# Add the parent directory (R4R) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from routing.soft_fusion_core import (
    load_sasrec_predictions,
    load_tiger_predictions,
    evaluate_fusion,
    grid_search_static_alpha,
    grid_search_dynamic,
    compute_single_model_baselines,
)


def plot_experiment_summary(
    dataset_label: str,
    normalization: str,
    top_k: int,
    static_curves: Dict[str, List[Tuple[float, float]]],
    grids: Dict[Tuple[str, str], pd.DataFrame],
    alpha_tests: Dict[Tuple[str, str], np.ndarray],
    output_path: str,
) -> None:
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    # Top row: separate static baseline plots for score and rank
    colors = {"score": "tab:blue", "rank": "tab:green"}

    ax_score = fig.add_subplot(gs[0, 0:2])
    curve_score = static_curves.get("score", [])
    if curve_score:
        xs, ys = zip(*sorted(curve_score))
        ax_score.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, color=colors["score"])
        best_idx = int(np.argmax(ys))
        ax_score.axvline(xs[best_idx], color=colors["score"], linestyle="--", alpha=0.5)
    ax_score.set_title(f"Static ensemble (score) — {dataset_label}")
    ax_score.set_xlabel("α")
    ax_score.set_ylabel(f"NDCG@{top_k}")
    ax_score.set_xlim(0, 1)
    ax_score.grid(True, alpha=0.3)

    ax_rank = fig.add_subplot(gs[0, 2:4])
    curve_rank = static_curves.get("rank", [])
    if curve_rank:
        xs, ys = zip(*sorted(curve_rank))
        ax_rank.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, color=colors["rank"])
        best_idx = int(np.argmax(ys))
        ax_rank.axvline(xs[best_idx], color=colors["rank"], linestyle="--", alpha=0.5)
    ax_rank.set_title(f"Static ensemble (rank) — {dataset_label}")
    ax_rank.set_xlabel("α")
    ax_rank.set_ylabel(f"NDCG@{top_k}")
    ax_rank.set_xlim(0, 1)
    ax_rank.grid(True, alpha=0.3)

    combos = [("score", "linear"), ("score", "sigmoid"), ("rank", "linear"), ("rank", "sigmoid")]
    for col, (fm, wf) in enumerate(combos):
        grid_df = grids.get((fm, wf))
        ax_hm = fig.add_subplot(gs[1, col])
        ax_hm.set_title(f"{fm}/{wf} — hyperparam search", fontsize=10)
        if grid_df is None or grid_df.empty:
            ax_hm.text(0.5, 0.5, "no data", ha="center", va="center")
            ax_hm.set_xticks([])
            ax_hm.set_yticks([])
        else:
            if wf == "sigmoid":
                p1, p2 = "k", "tau"
                xlabel, ylabel = "k", "τ"
            else:
                p1, p2 = "w", "b"
                xlabel, ylabel = "w", "b"
            if {p1, p2, "ndcg"} <= set(grid_df.columns):
                pivot = (
                    grid_df.pivot_table(values="ndcg", index=p2, columns=p1, aggfunc="mean")
                    .sort_index(axis=0)
                    .sort_index(axis=1)
                )
                xs = pivot.columns.to_numpy()
                ys = pivot.index.to_numpy()
                X, Y = np.meshgrid(xs, ys)
                im = ax_hm.pcolormesh(X, Y, pivot.values, cmap="viridis", shading="auto")
                for i, yv in enumerate(ys):
                    for j, xv in enumerate(xs):
                        val = pivot.values[i, j]
                        val_str = f"{val:.4f}"
                        if val_str.startswith("-0"):
                            val_str = "-" + val_str[2:]
                        elif val_str.startswith("0"):
                            val_str = val_str[1:]
                        ax_hm.text(xv, yv, val_str, ha="center", va="center", fontsize=7, color="white")
                ax_hm.set_xlabel(xlabel)
                ax_hm.set_ylabel(ylabel)
                cbar = plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
                cbar.set_label(f"NDCG@{top_k}")
            else:
                ax_hm.text(0.5, 0.5, "missing grid columns", ha="center", va="center")
                ax_hm.set_xticks([])
                ax_hm.set_yticks([])

        ax_hist = fig.add_subplot(gs[2, col])
        alphas = alpha_tests.get((fm, wf))
        ax_hist.set_title(f"{fm}/{wf} — α distribution", fontsize=10)
        if alphas is None or alphas.size == 0:
            ax_hist.text(0.5, 0.5, "no data", ha="center", va="center")
            ax_hist.set_xticks([])
            ax_hist.set_yticks([])
        else:
            ax_hist.hist(alphas, bins=20, range=(0, 1), color="steelblue", edgecolor="white", alpha=0.85)
            ax_hist.set_xlim(0, 1)
            ax_hist.set_xlabel("α")
            ax_hist.set_ylabel("count")
            mu, sigma = float(np.mean(alphas)), float(np.std(alphas))
            ax_hist.axvline(mu, color="red", linestyle="--", alpha=0.7)
            ax_hist.text(
                0.02,
                0.95,
                f"mean={mu:.2f}\nstd={sigma:.2f}",
                transform=ax_hist.transAxes,
                fontsize=8,
                va="top",
                ha="left",
            )

    fig.suptitle(f"Soft-fusion experiment — {dataset_label} (normalization={normalization})", fontsize=14, y=0.98)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Run soft-fusion experiment for one dataset")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--category", default="")
    p.add_argument("--version", default="")
    p.add_argument("--sas_val_csv", required=True)
    p.add_argument("--tiger_val_csv", required=True)
    p.add_argument("--sas_test_csv", required=True)
    p.add_argument("--tiger_test_csv", required=True)
    p.add_argument("--output_dir", default="routing/results")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_predictions", type=int, default=50)
    p.add_argument("--k_range", nargs=3, type=float, default=[1.0, 25.0, 4.0])
    p.add_argument("--tau_range", nargs=3, type=float, default=[0.0, 0.5, 0.1])
    p.add_argument("--w_range", nargs=3, type=float, default=[0.0, 10.0, 2.0])
    p.add_argument("--b_range", nargs=3, type=float, default=[0.0, 1.0, 0.2])
    p.add_argument("--alpha_range", nargs=3, type=float, default=[0.0, 1.0, 0.1])
    p.add_argument("--normalization", choices=["min_max", "standardize"], default="min_max")
    p.add_argument("--verbose", action="store_true", default=False)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_id = args.dataset_name
    if args.version:
        dataset_id = f"{args.dataset_name}-{args.version}"
    elif args.category:
        dataset_id = f"{args.dataset_name}-{args.category}"
    dataset_label = dataset_id

    sas_val = load_sasrec_predictions(args.sas_val_csv, "val", args.n_predictions)
    tig_val = load_tiger_predictions(args.tiger_val_csv, "val", args.n_predictions)
    sas_test = load_sasrec_predictions(args.sas_test_csv, "test", args.n_predictions)
    tig_test = load_tiger_predictions(args.tiger_test_csv, "test", args.n_predictions)

    sas_val_ndcg, tig_val_ndcg = compute_single_model_baselines(sas_val, tig_val, args.top_k)
    sas_test_ndcg, tig_test_ndcg = compute_single_model_baselines(sas_test, tig_test, args.top_k)

    models_rows = []

    models_rows.append(
        {
            "Dataset": dataset_label,
            "Model": "SASRec",
            "Fusion_method": "none",
            "Weighting_func": "none",
            "Normalization": args.normalization,
            "Best_params": "",
            "Test_ndcg@10": sas_test_ndcg,
            "Test_recall@10": np.nan,
            "Val_ndcg@10": sas_val_ndcg,
            "Improve over SASRec": 0.0,
            "Improve_over_TIGER": 0.0,
            "Improve_over_static": np.nan,
        }
    )
    models_rows.append(
        {
            "Dataset": dataset_label,
            "Model": "TIGER",
            "Fusion_method": "none",
            "Weighting_func": "none",
            "Normalization": args.normalization,
            "Best_params": "",
            "Test_ndcg@10": tig_test_ndcg,
            "Test_recall@10": np.nan,
            "Val_ndcg@10": tig_val_ndcg,
            "Improve over SASRec": (tig_test_ndcg - sas_test_ndcg) / sas_test_ndcg * 100 if sas_test_ndcg > 0 else 0.0,
            "Improve_over_TIGER": 0.0,
            "Improve_over_static": np.nan,
        }
    )

    static_curves: Dict[str, List[Tuple[float, float]]] = {}
    grids: Dict[Tuple[str, str], pd.DataFrame] = {}
    alpha_tests: Dict[Tuple[str, str], np.ndarray] = {}
    static_best: Dict[str, float] = {}
    static_test_ndcg: Dict[str, float] = {}

    alpha_min, alpha_max, alpha_step = args.alpha_range
    alphas = np.arange(alpha_min, alpha_max + 0.5 * alpha_step, alpha_step)

    k_min, k_max, k_step = args.k_range
    tau_min, tau_max, tau_step = args.tau_range
    w_min, w_max, w_step = args.w_range
    b_min, b_max, b_step = args.b_range
    k_vals = np.arange(k_min, k_max + 0.5 * k_step, k_step)
    tau_vals = np.arange(tau_min, tau_max + 0.5 * tau_step, tau_step)
    w_vals = np.arange(w_min, w_max + 0.5 * w_step, w_step)
    b_vals = np.arange(b_min, b_max + 0.5 * b_step, b_step)

    for fusion_method in ["score", "rank"]:
        print(f"[{dataset_label}] Static alpha sweep for fusion_method={fusion_method}")
        curve, best_alpha = grid_search_static_alpha(
            sas_val=sas_val,
            tiger_val=tig_val,
            alphas=alphas,
            fusion_method=fusion_method,
            normalization=args.normalization,
            top_k=args.top_k,
            verbose=args.verbose,
        )
        static_curves[fusion_method] = curve
        static_best[fusion_method] = best_alpha

        static_val_res = evaluate_fusion(
            sas_val,
            tig_val,
            top_k=args.top_k,
            fusion_method=fusion_method,
            normalization=args.normalization,
            weighting_mode="static",
            static_alpha=best_alpha,
        )
        static_test_res = evaluate_fusion(
            sas_test,
            tig_test,
            top_k=args.top_k,
            fusion_method=fusion_method,
            normalization=args.normalization,
            weighting_mode="static",
            static_alpha=best_alpha,
        )
        static_test_ndcg[fusion_method] = static_test_res["avg_ndcg"]

        models_rows.append(
            {
                "Dataset": dataset_label,
                "Model": f"Static Fusion ({fusion_method})",
                "Fusion_method": fusion_method,
                "Weighting_func": "static",
                "Normalization": args.normalization,
                "Best_params": f"alpha={best_alpha:.2f}",
                "Test_ndcg@10": static_test_res["avg_ndcg"],
                "Test_recall@10": static_test_res["avg_recall"],
                "Val_ndcg@10": static_val_res["avg_ndcg"],
                "Improve over SASRec": (static_test_res["avg_ndcg"] - sas_test_ndcg) / sas_test_ndcg * 100 if sas_test_ndcg > 0 else 0.0,
                "Improve_over_TIGER": (static_test_res["avg_ndcg"] - tig_test_ndcg) / tig_test_ndcg * 100 if tig_test_ndcg > 0 else 0.0,
                "Improve_over_static": 0.0,
            }
        )

        for weighting_fn in ["linear", "sigmoid"]:
            print(f"[{dataset_label}] Grid search for fusion={fusion_method}, weighting_fn={weighting_fn}")
            if weighting_fn == "sigmoid":
                gs_res = grid_search_dynamic(
                    sas_val=sas_val,
                    tiger_val=tig_val,
                    weighting_fn="sigmoid",
                    p1_values=k_vals,
                    p2_values=tau_vals,
                    fusion_method=fusion_method,
                    normalization=args.normalization,
                    top_k=args.top_k,
                    verbose=args.verbose,
                )
                grids[(fusion_method, weighting_fn)] = gs_res["grid_table"]
                best_k, best_tau = gs_res["best_params"]
                val_res = evaluate_fusion(
                    sas_val,
                    tig_val,
                    top_k=args.top_k,
                    fusion_method=fusion_method,
                    normalization=args.normalization,
                    weighting_mode="dynamic",
                    weighting_fn="sigmoid",
                    k_steepness=best_k,
                    tau_threshold=best_tau,
                )
                test_res = evaluate_fusion(
                    sas_test,
                    tig_test,
                    top_k=args.top_k,
                    fusion_method=fusion_method,
                    normalization=args.normalization,
                    weighting_mode="dynamic",
                    weighting_fn="sigmoid",
                    k_steepness=best_k,
                    tau_threshold=best_tau,
                )
                alpha_tests[(fusion_method, weighting_fn)] = np.array([d["alpha"] for d in test_res["details"]], dtype=float)
                best_params_str = f"k={best_k:.2f},tau={best_tau:.2f}"
            else:
                gs_res = grid_search_dynamic(
                    sas_val=sas_val,
                    tiger_val=tig_val,
                    weighting_fn="linear",
                    p1_values=w_vals,
                    p2_values=b_vals,
                    fusion_method=fusion_method,
                    normalization=args.normalization,
                    top_k=args.top_k,
                    verbose=args.verbose,
                )
                grids[(fusion_method, weighting_fn)] = gs_res["grid_table"]
                best_w, best_b = gs_res["best_params"]
                val_res = evaluate_fusion(
                    sas_val,
                    tig_val,
                    top_k=args.top_k,
                    fusion_method=fusion_method,
                    normalization=args.normalization,
                    weighting_mode="dynamic",
                    weighting_fn="linear",
                    w=best_w,
                    b=best_b,
                )
                test_res = evaluate_fusion(
                    sas_test,
                    tig_test,
                    top_k=args.top_k,
                    fusion_method=fusion_method,
                    normalization=args.normalization,
                    weighting_mode="dynamic",
                    weighting_fn="linear",
                    w=best_w,
                    b=best_b,
                )
                alpha_tests[(fusion_method, weighting_fn)] = np.array([d["alpha"] for d in test_res["details"]], dtype=float)
                best_params_str = f"w={best_w:.2f},b={best_b:.2f}"

            static_ref = static_test_ndcg[fusion_method]

            print(
                f"[{dataset_label}] Best ({fusion_method},{weighting_fn}) params={best_params_str} "
                f"val_ndcg={val_res['avg_ndcg']:.4f} test_ndcg={test_res['avg_ndcg']:.4f}"
            )

            models_rows.append(
                {
                    "Dataset": dataset_label,
                    "Model": f"Dynamic Fusion ({fusion_method}+{weighting_fn})",
                    "Fusion_method": fusion_method,
                    "Weighting_func": weighting_fn,
                    "Normalization": args.normalization,
                    "Best_params": best_params_str,
                    "Test_ndcg@10": test_res["avg_ndcg"],
                    "Test_recall@10": test_res["avg_recall"],
                    "Val_ndcg@10": val_res["avg_ndcg"],
                    "Improve over SASRec": (test_res["avg_ndcg"] - sas_test_ndcg) / sas_test_ndcg * 100 if sas_test_ndcg > 0 else 0.0,
                    "Improve_over_TIGER": (test_res["avg_ndcg"] - tig_test_ndcg) / tig_test_ndcg * 100 if tig_test_ndcg > 0 else 0.0,
                    "Improve_over_static": (test_res["avg_ndcg"] - static_ref) / static_ref * 100 if static_ref > 0 else 0.0,
                }
            )

            print(
                f"[{dataset_label}] ({fusion_method},{weighting_fn}) "
                f"improve_vs_SASRec={models_rows[-1]['Improve over SASRec']:.2f}% "
                f"improve_vs_TIGER={models_rows[-1]['Improve_over_TIGER']:.2f}% "
                f"improve_vs_static={models_rows[-1]['Improve_over_static']:.2f}%"
            )

    summary_df = pd.DataFrame(models_rows)
    csv_path = os.path.join(args.output_dir, f"summary_{dataset_label}.csv")
    summary_df.to_csv(csv_path, index=False)

    fig_path = os.path.join(args.output_dir, f"figure_{dataset_label}.png")
    plot_experiment_summary(
        dataset_label=dataset_label,
        normalization=args.normalization,
        top_k=args.top_k,
        static_curves=static_curves,
        grids=grids,
        alpha_tests=alpha_tests,
        output_path=fig_path,
    )

    if args.verbose:
        print("\n================ Hyperparameter Tuning Summary ================")
        print(f"Dataset: {dataset_label}")
        print(f"Normalization: {args.normalization}")
        print(f"SASRec test NDCG@{args.top_k}: {sas_test_ndcg:.4f}")
        print(f"TIGER  test NDCG@{args.top_k}: {tig_test_ndcg:.4f}")
        print("")
        for fm in ["score", "rank"]:
            for wf in ["static", "linear", "sigmoid"]:
                if wf == "static":
                    model_name = f"Static Fusion ({fm})"
                else:
                    model_name = f"Dynamic Fusion ({fm}+{wf})"
                rows = summary_df[summary_df["Model"] == model_name]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                print(
                    f"[{fm:5s}, {wf:7s}] "
                    f"test_ndcg={row['Test_ndcg@10']:.4f} "
                    f"val_ndcg={row['Val_ndcg@10']:.4f} "
                    f"ΔSASRec={row['Improve over SASRec']:.2f}% "
                    f"ΔTIGER={row['Improve_over_TIGER']:.2f}% "
                    f"Δstatic={row['Improve_over_static']:.2f}% "
                    f"params={row['Best_params']}"
                )
        print("===============================================================")

    print(f"Summary saved to {csv_path}")
    print(f"Figure saved to {fig_path}")
    print("===============================================================")


if __name__ == "__main__":
    main()

