#!/usr/bin/env python3

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple


def parse_specs(raw: List[str]) -> List[Tuple[str, str, str]]:
    """
    Each element in raw is a dataset_id string like:
      - AmazonReviews2014-Sports_and_Outdoors
      - Yelp-Yelp_2020

    We split at the first '-' to get dataset_name and the rest.
    If dataset_name contains 'Yelp', the suffix is treated as version; otherwise as category.
    """
    specs: List[Tuple[str, str, str]] = []
    for s in raw:
        s = s.strip()
        if "-" in s:
            name, suffix = s.split("-", 1)
            if "Yelp" in name:
                specs.append((name, "", suffix))
            else:
                specs.append((name, suffix, ""))
        else:
            specs.append((s, "", ""))
    return specs


def run_one(
    python_path: str,
    dataset_name: str,
    category: str,
    version: str,
    base_dir: str,
    output_dir: str,
    top_k: int,
    n_predictions: int,
    normalization: str,
    k_range: List[float],
    tau_range: List[float],
    w_range: List[float],
    b_range: List[float],
    alpha_range: List[float],
) -> str:
    if version:
        ds_id = f"{dataset_name}-{version}"
    elif category:
        ds_id = f"{dataset_name}-{category}"
    else:
        ds_id = dataset_name
    sas_val = os.path.join(base_dir, "SASRec", "results", f"sasrec_predictions_with_scores_val_{ds_id}.csv")
    tig_val = os.path.join(base_dir, "TIGER", "results", f"tiger_predictions_with_scores_val_{ds_id}.csv")
    sas_tes = os.path.join(base_dir, "SASRec", "results", f"sasrec_predictions_with_scores_test_{ds_id}.csv")
    tig_tes = os.path.join(base_dir, "TIGER", "results", f"tiger_predictions_with_scores_test_{ds_id}.csv")

    cmd = [
        python_path,
        "routing/run_soft_fusion_experiment.py",
        "--dataset_name",
        dataset_name,
        "--category",
        category,
        "--version",
        version,
        "--sas_val_csv",
        sas_val,
        "--tiger_val_csv",
        tig_val,
        "--sas_test_csv",
        sas_tes,
        "--tiger_test_csv",
        tig_tes,
        "--output_dir",
        output_dir,
        "--top_k",
        str(top_k),
        "--n_predictions",
        str(n_predictions),
        "--normalization",
        normalization,
        "--k_range",
        *[str(x) for x in k_range],
        "--tau_range",
        *[str(x) for x in tau_range],
        "--w_range",
        *[str(x) for x in w_range],
        "--b_range",
        *[str(x) for x in b_range],
        "--alpha_range",
        *[str(x) for x in alpha_range],
    ]
    subprocess.run(cmd, check=True)
    return ds_id


def main() -> None:
    p = argparse.ArgumentParser(description="Run soft-fusion experiments on multiple datasets in parallel")
    p.add_argument("--python_path", default="/home/USER/miniconda3/envs/GenRec/bin/python")
    p.add_argument("--base_dir", default="Confidence")
    p.add_argument("--output_dir", default="routing/results")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_predictions", type=int, default=50)
    p.add_argument("--normalization", choices=["min_max", "standardize"], default="min_max")
    p.add_argument("--max_workers", type=int, default=2)

    p.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "AmazonReviews2014-Sports_and_Outdoors",
            "AmazonReviews2014-Beauty",
            "AmazonReviews2023-Office_Products",
            "AmazonReviews2023-Industrial_and_Scientific",
            "AmazonReviews2023-Musical_Instruments",
            "Steam",
            "Yelp-Yelp_2020",
        ],
        help='Datasets as "DatasetID", e.g. "AmazonReviews2014-Sports_and_Outdoors" or "Yelp-Yelp_2020"',
    )

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
                run_one,
                args.python_path,
                dataset_name,
                category,
                version,
                args.base_dir,
                args.output_dir,
                args.top_k,
                args.n_predictions,
                args.normalization,
                args.k_range,
                args.tau_range,
                args.w_range,
                args.b_range,
                args.alpha_range,
            ): f"{dataset_name}-{version}" if version else f"{dataset_name}-{category}" if category else dataset_name
            for (dataset_name, category, version) in specs
        }

        for fut in as_completed(futures):
            fut.result()


if __name__ == "__main__":
    main()

