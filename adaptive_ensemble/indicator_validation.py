#!/usr/bin/env python3
"""
Section 5.2: MSP Indicator Validation
Validate that MSP correlates with memorization labels and model performance.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

def build_msp_bins(sas_csv, tiger_csv, labels_json, n_bins=5):
    df_sas = pd.read_csv(sas_csv)
    df_tig = pd.read_csv(tiger_csv)
    df_tig = df_tig[df_tig['beam_rank'] == 0].copy()

    if 'sample_idx' in df_sas.columns:
        df_sas.rename(columns={'sample_idx': 'sample_id'}, inplace=True)

    with open(labels_json) as f:
        labels = json.load(f)

    df = pd.merge(
        df_sas[['sample_id', 'confidence_msp', 'ndcg@10']].rename(
            columns={'ndcg@10': 'sas_ndcg'}),
        df_tig[['sample_id', 'ndcg@10']].rename(
            columns={'ndcg@10': 'tiger_ndcg'}),
        on='sample_id',
    )

    df['is_memo'] = df['sample_id'].apply(
        lambda x: int('memorization' in labels.get(str(int(x)), [])))

    df['bin'] = pd.qcut(
        df['confidence_msp'].rank(method='first'), q=n_bins, labels=False)

    df_bins = (
        df.groupby('bin')
        .agg(
            msp_mean=('confidence_msp', 'mean'),
            msp_min=('confidence_msp', 'min'),
            msp_max=('confidence_msp', 'max'),
            memo_count=('is_memo', 'sum'),
            memo_ratio=('is_memo', 'mean'),
            sas_ndcg=('sas_ndcg', 'mean'),
            tiger_ndcg=('tiger_ndcg', 'mean'),
            count=('sample_id', 'count'),
        )
        .reset_index()
    )
    df_bins['memo_ratio_overall'] = df['is_memo'].mean()
    return df_bins


def print_summary(dataset_id, split, df_bins):
    overall = df_bins['memo_ratio_overall'].iloc[0]
    print(f"\n  MSP Indicator Validation — {dataset_id} ({split})")
    print(f"{'=' * 72}")
    print(f"  Overall memorization ratio: {overall:.4f} ({overall * 100:.1f}%)")
    print(f"  Bins: {len(df_bins)}\n")
    header = (f"  {'Bin':>3}  {'MSP Range':>16}  {'Memo%':>7}  "
              f"{'SASRec':>8}  {'TIGER':>8}  {'N':>6}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for _, r in df_bins.iterrows():
        print(f"  {int(r['bin']):>3}  "
              f"[{r['msp_min']:.4f}, {r['msp_max']:.4f}]  "
              f"{r['memo_ratio'] * 100:>6.1f}%  "
              f"{r['sas_ndcg']:>8.4f}  {r['tiger_ndcg']:>8.4f}  "
              f"{int(r['count']):>6}")
    print(f"{'=' * 72}")


def main():
    p = argparse.ArgumentParser(description="Validate MSP as an indicator for adaptive ensemble routing")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Dataset IDs, e.g. AmazonReviews2014-Sports_and_Outdoors")
    p.add_argument("--split", default="test")
    p.add_argument("--labels_dir", default="outputs",
                   help="Directory containing item_case_labels JSON files")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--n_bins", type=int, default=5)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_bins = []

    for dataset_id in args.datasets:
        sas_csv = os.path.join(
            f"outputs/sasrec_predictions_with_scores_{args.split}_{dataset_id}.csv")
        tiger_csv = os.path.join(
            f"outputs/tiger_predictions_with_scores_{args.split}_{dataset_id}.csv")
        labels_json = os.path.join(
            args.labels_dir,
            f"token_mem_{dataset_id}_{args.split}_item_case_labels.json")

        for path in [sas_csv, tiger_csv, labels_json]:
            if not os.path.exists(path):
                print(f"[Skip] Missing: {path}")
                break
        else:
            df_bins = build_msp_bins(sas_csv, tiger_csv, labels_json,
                                     n_bins=args.n_bins)
            print_summary(dataset_id, args.split, df_bins)

            out_path = os.path.join(
                args.output_dir,
                f"msp_indicator_{dataset_id}_{args.split}.csv")
            df_bins.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}\n")

            df_bins_tagged = df_bins.copy()
            df_bins_tagged['dataset'] = dataset_id
            all_bins.append(df_bins_tagged)

    if len(all_bins) > 1:
        combined = pd.concat(all_bins, ignore_index=True)
        combined_path = os.path.join(
            args.output_dir, f"msp_indicator_combined_{args.split}.csv")
        combined.to_csv(combined_path, index=False)
        print(f"  Combined results saved: {combined_path}")

        print(f"\n  Cross-Dataset Summary ({args.split})")
        print(f"{'=' * 60}")
        print(f"  {'Dataset':<45} {'Memo%':>7}  {'Corr':>6}")
        print(f"  {'-' * 56}")
        for dataset_id in combined['dataset'].unique():
            sub = combined[combined['dataset'] == dataset_id]
            overall = sub['memo_ratio_overall'].iloc[0]
            corr = np.corrcoef(sub['bin'], sub['memo_ratio'])[0, 1]
            print(f"  {dataset_id:<45} {overall * 100:>6.1f}%  {corr:>6.2f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
