"""
Section 4.3: Explaining Performance Trade-off via Token Memorization
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genrec.pipeline import Pipeline

# Helpers
def load_precomputed(out_prefix):
    with open(f"{out_prefix}_meta.json") as f:
        meta = json.load(f)

    with open(f"{out_prefix}_item_case_labels.json") as f:
        raw = json.load(f)
    item_case_labels = {int(k): set(v) for k, v in raw.items()}

    with open(f"{out_prefix}_token_case_labels.json") as f:
        raw = json.load(f)
    token_case_labels = {
        int(k): {int(pk): set(pv) for pk, pv in v.items()}
        for k, v in raw.items()
    }

    df_clean = pd.read_csv(f"{out_prefix}_df_clean.csv")
    return meta, item_case_labels, token_case_labels, df_clean


def is_item_generalization(item_labels):
    if 'memorization' in item_labels:
        return False
    sub_cats = ['substitutability', 'symmetry', 'transitivity', '2nd-symmetry', 'uncategorized']
    return any(label.split('_')[0] in sub_cats for label in item_labels)


def item_prefix(item, k, tokenizer):
    toks = tokenizer._token_single_item(item)
    if not isinstance(toks, (list, tuple)):
        toks = [toks]
    if not toks:
        return None
    return tuple(toks[:k]) if len(toks) >= k else tuple(toks)

# Prefix transition statistics
def build_prefix_transition_counts(train_item_seqs, depths, k_window, tokenizer):
    num_counts = defaultdict(int)
    denom_counts = defaultdict(int)

    for seq in tqdm(train_item_seqs, desc="Prefix transition stats"):
        n = len(seq)
        if n < 2:
            continue
        for i in range(n):
            u = seq[i]
            for offset in range(1, k_window + 1):
                j = i + offset
                if j >= n:
                    break
                v = seq[j]
                for k in depths:
                    pu = item_prefix(u, k, tokenizer)
                    pv = item_prefix(v, k, tokenizer)
                    if pu and pv:
                        num_counts[(k, pu, pv)] += 1
                        denom_counts[(k, pu)] += 1

    return num_counts, denom_counts


def build_df_pred(df_clean, test_item_seqs, depths, num_counts, denom_counts, tokenizer):
    df_gen = df_clean[df_clean['is_item_generalization']].copy()
    rows = []
    for _, row in df_gen.iterrows():
        sample_id = row['sample_id']
        item_seq = test_item_seqs[sample_id]
        if not item_seq or len(item_seq) < 2:
            continue
        u, v = item_seq[-2], item_seq[-1]
        ndcg_t, ndcg_s = row['ndcg@10_tiger'], row['ndcg@10_sasrec']

        for k in depths:
            pu, pv = item_prefix(u, k, tokenizer), item_prefix(v, k, tokenizer)
            ratio = 0.0
            if pu and pv:
                den = denom_counts.get((k, pu), 0)
                if den > 0:
                    ratio = num_counts.get((k, pu, pv), 0) / den
            rows.append({
                'prefix_len': k,
                'predictability': ratio,
                'ndcg_tiger': ndcg_t,
                'ndcg_sasrec': ndcg_s,
            })
    return pd.DataFrame(rows)


def build_df_raw(df_clean, test_item_seqs, depths, num_counts, tokenizer):
    df_gen = df_clean[df_clean['is_item_generalization']].copy()
    rows = []
    for _, row in df_gen.iterrows():
        sample_id = row['sample_id']
        item_seq = test_item_seqs[sample_id]
        if not item_seq or len(item_seq) < 2:
            continue
        u, v = item_seq[-2], item_seq[-1]
        ndcg_t, ndcg_s = row['ndcg@10_tiger'], row['ndcg@10_sasrec']

        for k in depths:
            pu, pv = item_prefix(u, k, tokenizer), item_prefix(v, k, tokenizer)
            raw_count = 0
            if pu and pv:
                raw_count = num_counts.get((k, pu, pv), 0)
            rows.append({
                'prefix_len': k,
                'raw_count': raw_count,
                'ndcg_tiger': ndcg_t,
                'ndcg_sasrec': ndcg_s,
            })
    return pd.DataFrame(rows)

# Purity statistics
def build_purity_counts(train_item_seqs, target_k, tokenizer):
    item_trans = defaultdict(int)
    item_ctx = defaultdict(int)
    prefix_trans = defaultdict(int)
    prefix_ctx = defaultdict(int)

    for seq in tqdm(train_item_seqs, desc="Purity stats"):
        n = len(seq)
        if n < 2:
            continue
        for i in range(n - 1):
            u, v = seq[i], seq[i + 1]
            item_trans[(u, v)] += 1
            item_ctx[u] += 1
            pu = item_prefix(u, target_k, tokenizer)
            pv = item_prefix(v, target_k, tokenizer)
            if pu and pv:
                prefix_trans[(target_k, pu, pv)] += 1
                prefix_ctx[(target_k, pu)] += 1

    return item_trans, item_ctx, prefix_trans, prefix_ctx


def build_df_heat(df_clean, test_item_seqs, target_k, tokenizer,
                  item_trans, item_ctx, prefix_trans, prefix_ctx):
    df_mem = df_clean[~df_clean['is_item_generalization']].copy()
    rows = []
    for _, row in df_mem.iterrows():
        sample_id = row['sample_id']
        item_seq = test_item_seqs[sample_id]
        if not item_seq or len(item_seq) < 2:
            continue
        u, v = item_seq[-2], item_seq[-1]

        i_t = item_trans.get((u, v), 0)
        i_c = item_ctx.get(u, 0)
        i_purity = i_t / i_c if i_c > 0 else 0.0

        pu = item_prefix(u, target_k, tokenizer)
        pv = item_prefix(v, target_k, tokenizer)
        p_purity = 0.0
        if pu and pv:
            p_t = prefix_trans.get((target_k, pu, pv), 0)
            p_c = prefix_ctx.get((target_k, pu), 0)
            p_purity = p_t / p_c if p_c > 0 else 0.0

        rows.append({
            'item_purity': i_purity,
            'prefix_purity': p_purity,
            'delta': row['ndcg@10_tiger'] - row['ndcg@10_sasrec'],
        })
    return pd.DataFrame(rows)

# MSP bins
def build_df_msp_bins(msp_path, sasrec_infer_path, tiger_infer_path, n_bins=4):
    df_msp = pd.read_csv(msp_path)
    df_sas = pd.read_csv(sasrec_infer_path)
    df_tig = pd.read_csv(tiger_infer_path)

    if 'beam_rank' in df_tig.columns:
        df_tig = df_tig[df_tig['beam_rank'] == 0].copy()

    for tmp in [df_msp, df_sas, df_tig]:
        if 'sample_idx' in tmp.columns:
            tmp.rename(columns={'sample_idx': 'sample_id'}, inplace=True)

    rank_col = 'rank_id' if 'rank_id' in df_sas.columns else 'rank'
    df_sas_top = df_sas[df_sas[rank_col] == 0].copy()
    rank_col = 'rank_id' if 'rank_id' in df_tig.columns else 'rank'
    df_tig_top = df_tig[df_tig[rank_col] == 0].copy()

    merge_cols_sas = ['sample_id', 'ndcg@10']
    if 'item_labels' in df_sas_top.columns:
        merge_cols_sas.append('item_labels')
    elif 'logic_label' in df_sas_top.columns:
        merge_cols_sas.append('logic_label')

    df_m = pd.merge(df_msp[['sample_id', 'mem_score']],
                    df_sas_top[merge_cols_sas], on='sample_id')
    df_m = pd.merge(
        df_m,
        df_tig_top[['sample_id', 'ndcg@10']],
        on='sample_id', suffixes=('_sas', '_tiger'),
    )

    if 'logic_label' in df_m.columns:
        df_m['is_memo'] = (df_m['logic_label'] == 'memorization').astype(int)
    elif 'item_labels' in df_m.columns:
        df_m['is_memo'] = df_m['item_labels'].apply(
            lambda x: 'memorization' in str(x)).astype(int)
    else:
        raise ValueError("No logic_label or item_labels column found")

    df_m['bin'] = pd.qcut(df_m['mem_score'].rank(method='first'), q=n_bins, labels=False)

    df_bins = (
        df_m.groupby('bin')
        .agg(
            score_mean=('mem_score', 'mean'),
            score_min=('mem_score', 'min'),
            score_max=('mem_score', 'max'),
            memo_count=('is_memo', 'sum'),
            memo_ratio=('is_memo', 'mean'),
            sas_ndcg=('ndcg@10_sas', 'mean'),
            tiger_ndcg=('ndcg@10_tiger', 'mean'),
            sample_count=('sample_id', 'count'),
        )
        .reset_index()
    )
    df_bins['memo_ratio_overall'] = df_m['is_memo'].mean()
    return df_bins

# Report
def print_generalization_table(df_raw, depths, n_count_bins=5):
    """TIGER vs SASRec NDCG@10 by token memorization support Cn(u, it)."""
    print(f"\n  --- Item Generalization: Performance vs Token Memorization Support Cn ---")

    for k in sorted(depths):
        sub = df_raw[df_raw['prefix_len'] == k].copy()
        if sub.empty:
            continue

        df_zero = sub[sub['raw_count'] == 0].copy()
        df_pos = sub[sub['raw_count'] > 0].copy()
        df_zero['bin_idx'] = 0
        df_zero['bin_label'] = "0"

        if len(df_pos) > 0:
            df_pos['bin_code'], bins = pd.qcut(
                df_pos['raw_count'], q=n_count_bins,
                duplicates='drop', retbins=True, labels=False)
            unique_codes = sorted(df_pos['bin_code'].unique())
            code_to_idx = {c: i + 1 for i, c in enumerate(unique_codes)}
            df_pos['bin_idx'] = df_pos['bin_code'].map(code_to_idx)
            df_pos['bin_label'] = df_pos['bin_code'].apply(
                lambda c: f"{int((bins[int(c)] + bins[int(c)+1]) / 2)}")
            df_all = pd.concat([df_zero, df_pos], ignore_index=True)
        else:
            df_all = df_zero

        agg = (df_all.groupby(['bin_idx', 'bin_label'])
               .agg(n=('raw_count', 'size'),
                    tiger=('ndcg_tiger', 'mean'),
                    sasrec=('ndcg_sasrec', 'mean'))
               .reset_index().sort_values('bin_idx'))

        print(f"\n  [n={k}] ({len(sub)} instances)")
        print(f"    {'Cn':>8}  {'N':>6}  {'TIGER':>8}  {'SASRec':>8}  {'Δ':>8}")
        print(f"    {'─'*8}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
        for _, r in agg.iterrows():
            delta = r['tiger'] - r['sasrec']
            print(f"    {r['bin_label']:>8}  {int(r['n']):>6}"
                  f"  {r['tiger']:>8.4f}  {r['sasrec']:>8.4f}  {delta:>+8.4f}")


def print_memorization_table(df_heat, n_bins=4):
    """Δ NDCG (TIGER - SASRec) by item transition prob φ × prefix transition prob ψ."""
    if df_heat.empty:
        return
    print(f"\n  --- Item Memorization: Δ NDCG by φ (item) × ψ (prefix) ---")

    df = df_heat.copy()
    try:
        df['xbin'], xb = pd.qcut(df['prefix_purity'], q=n_bins,
                                  duplicates='drop', labels=False, retbins=True)
    except ValueError:
        df['xbin'], xb = pd.cut(df['prefix_purity'], bins=n_bins,
                                labels=False, retbins=True)
    try:
        df['ybin'], yb = pd.qcut(df['item_purity'], q=n_bins,
                                 duplicates='drop', labels=False, retbins=True)
    except ValueError:
        df['ybin'], yb = pd.cut(df['item_purity'], bins=n_bins,
                                labels=False, retbins=True)

    pv = df.pivot_table(index='ybin', columns='xbin', values='delta', aggfunc='mean')
    ct = df.pivot_table(index='ybin', columns='xbin', values='delta', aggfunc='size')

    nx, ny = len(xb) - 1, len(yb) - 1
    pv = pv.reindex(columns=np.arange(nx)).reindex(index=np.arange(ny)[::-1])
    ct = ct.reindex(columns=np.arange(nx)).reindex(index=np.arange(ny)[::-1])

    x_labels = [f"[{xb[i]:.2f},{xb[i+1]:.2f}]" for i in range(nx)]
    y_labels = [f"[{yb[i]:.2f},{yb[i+1]:.2f}]" for i in range(ny)][::-1]

    col_w = max(len(l) for l in x_labels) + 2
    row_w = max(len(l) for l in y_labels) + 2

    print(f"\n  ψ = prefix transition prob (cols) × φ = item transition prob (rows)")
    print(f"  Δ NDCG = TIGER - SASRec")
    print(f"  {len(df)} memorization instances, {n_bins}×{n_bins} grid\n")

    header = " " * (row_w + 2) + "".join(f"{l:>{col_w}}" for l in x_labels)
    print(f"  {'ψ →':>{row_w}}  {header.strip()}")
    print(f"  {'φ ↓':>{row_w}}  {'─' * (col_w * nx)}")

    for yi in range(ny):
        cells = []
        for xi in range(nx):
            v = pv.iloc[yi, xi]
            c = ct.iloc[yi, xi]
            if pd.isna(v) or pd.isna(c) or c == 0:
                cells.append(f"{'—':>{col_w}}")
            else:
                cells.append(f"{v:>+{col_w-1}.3f} " if col_w > 7
                             else f"{v:>+.3f}")
        print(f"  {y_labels[yi]:>{row_w}}  {''.join(cells)}")

    print(f"  {' ' * row_w}  {'─' * (col_w * nx)}")
    print(f"  {'N':>{row_w}}  " +
          "".join(f"{int(ct.iloc[:, xi].sum()):>{col_w}}" for xi in range(nx)))

    print(f"\n  Overall Δ NDCG: {df['delta'].mean():+.4f} ({len(df)} instances)")


def print_summary(df_raw, df_heat, depths, target_k, dataset_id, n_count_bins=5):
    print(f"\n{'=' * 70}")
    print(f"  Performance Analysis — {dataset_id}")
    print(f"{'=' * 70}")

    print_generalization_table(df_raw, depths, n_count_bins)
    print_memorization_table(df_heat)

    print(f"{'=' * 70}\n")

def main():
    p = argparse.ArgumentParser(description="Performance analysis (prefix transitions)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--category", default=None)
    p.add_argument("--version", default=None)
    p.add_argument("--sem_ids_path", required=True)
    p.add_argument("--sasrec_infer_path", required=True)
    p.add_argument("--tiger_infer_path", required=True)
    p.add_argument("--msp_path", default=None,
                   help="Path to MSP mapping CSV (optional)")
    p.add_argument("--split", default="test")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--n_msp_bins", type=int, default=4)
    args = p.parse_args()

    dataset_id = args.dataset
    if args.version:
        dataset_id = f"{args.dataset}-{args.version}"
    elif args.category:
        dataset_id = f"{args.dataset}-{args.category}"

    out_prefix = os.path.join(args.output_dir, f"token_mem_{dataset_id}_{args.split}")

    # --- Load pre-computed support coverage data ---
    print(f"Loading pre-computed data from {out_prefix}_*")
    meta, item_case_labels, token_case_labels, df_clean = load_precomputed(out_prefix)
    depths = meta['DEPTHS']
    k_window = meta['K_WINDOW']
    target_k = meta['TARGET_K']

    # --- Load TIGER pipeline for tokenizer ---
    config_tiger = {'logging': False, 'sem_ids_path': args.sem_ids_path}
    if args.category:
        config_tiger['category'] = args.category
    if args.version:
        config_tiger['version'] = args.version

    print("Loading TIGER pipeline...")
    tiger_pipeline = Pipeline(model_name='TIGER', dataset_name=args.dataset,
                              config_dict=config_tiger)
    tokenizer = tiger_pipeline.tokenizer
    train_item_seqs = tiger_pipeline.split_datasets['train']['item_seq']
    test_item_seqs = tiger_pipeline.split_datasets[args.split]['item_seq']

    # --- Prefix transition counts ---
    print("Building prefix transition counts...")
    num_counts, denom_counts = build_prefix_transition_counts(
        train_item_seqs, depths, k_window, tokenizer)

    # --- Build analysis dataframes ---
    print("Building df_pred...")
    df_pred = build_df_pred(df_clean, test_item_seqs, depths,
                            num_counts, denom_counts, tokenizer)

    print("Building df_raw...")
    df_raw = build_df_raw(df_clean, test_item_seqs, depths, num_counts, tokenizer)

    print("Building purity statistics...")
    it, ic, pt, pc = build_purity_counts(train_item_seqs, target_k, tokenizer)

    print("Building df_heat...")
    df_heat = build_df_heat(df_clean, test_item_seqs, target_k, tokenizer,
                            it, ic, pt, pc)

    # --- Save ---
    df_pred.to_csv(f"{out_prefix}_df_pred.csv", index=False)
    df_raw.to_csv(f"{out_prefix}_df_raw.csv", index=False)
    df_heat.to_csv(f"{out_prefix}_df_heat.csv", index=False)

    # --- MSP bins (optional) ---
    df_msp_bins = None
    if args.msp_path and os.path.exists(args.msp_path):
        print("Building MSP bins...")
        df_msp_bins = build_df_msp_bins(
            args.msp_path, args.sasrec_infer_path, args.tiger_infer_path,
            n_bins=args.n_msp_bins)
        df_msp_bins.to_csv(f"{out_prefix}_df_msp_bins.csv", index=False)

    print(f"All results saved to {out_prefix}_*")

    # --- Report ---
    print_summary(df_raw, df_heat, depths, target_k, dataset_id)


if __name__ == "__main__":
    main()
