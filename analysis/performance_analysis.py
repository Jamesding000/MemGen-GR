"""
Section 4.3: Explaining Performance Trade-off via Token Memorization
"""

import argparse
import json
import os
import sys
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from genrec.pipeline import Pipeline

def compute_transition_stats(train_seqs, tokenizer, depths, target_k, k_window):
    """Single-pass computation of all item and prefix transition frequencies."""
    
    @lru_cache(maxsize=100000)
    def get_prefix(item, k):
        toks = tokenizer._token_single_item(item)
        toks = [toks] if not isinstance(toks, (list, tuple)) else toks
        return tuple(toks[:k]) if len(toks) >= k else tuple(toks) if toks else None

    # memorization metrics
    i_ctx, i_trans = Counter(), Counter()
    p_ctx_strict, p_trans_strict = Counter(), Counter()
    
    # generalization metrics
    p_trans_win = Counter()

    for seq in tqdm(train_seqs, desc="Computing transition stats"):
        n = len(seq)
        for i in range(n - 1):
            u = seq[i]
            nxt = seq[i + 1]
            
            # 1-hop transition counts
            i_ctx[u] += 1
            i_trans[(u, nxt)] += 1
            
            pu_t, pv_t = get_prefix(u, target_k), get_prefix(nxt, target_k)
            if pu_t and pv_t:
                p_ctx_strict[pu_t] += 1
                p_trans_strict[(pu_t, pv_t)] += 1

            # k-hop transition counts
            for j in range(i + 1, min(i + 1 + k_window, n)):
                v = seq[j]
                for k in depths:
                    pu, pv = get_prefix(u, k), get_prefix(v, k)
                    if pu and pv:
                        p_trans_win[(k, pu, pv)] += 1

    return (i_ctx, i_trans, p_ctx_strict, p_trans_strict, p_trans_win), get_prefix


def build_analysis_dfs(df_clean, test_seqs, stats, depths, target_k, get_prefix):
    """Vectorized construction of Generalization and Memorization DataFrames."""
    i_ctx, i_trans, p_ctx_strict, p_trans_strict, p_trans_win = stats
    gen_rows, mem_rows = [], []

    for _, r in df_clean.iterrows():
        seq = test_seqs[int(r['sample_id'])]
        if len(seq) < 2: 
            continue
            
        u, v = seq[-2], seq[-1]
        t_ndcg, s_ndcg = r['ndcg@10_tiger'], r['ndcg@10_sasrec']
        delta = t_ndcg - s_ndcg

        # generalization support
        if r['is_item_generalization']:
            for k in depths:
                pu, pv = get_prefix(u, k), get_prefix(v, k)
                count = p_trans_win.get((k, pu, pv), 0) if (pu and pv) else 0
                gen_rows.append({'k': k, 'count': count, 'tiger': t_ndcg, 'sasrec': s_ndcg})
                
        # memorization purity
        else:
            ic = i_ctx.get(u, 0)
            pu, pv = get_prefix(u, target_k), get_prefix(v, target_k)
            pc = p_ctx_strict.get(pu, 0) if pu else 0

            mem_rows.append({
                'item_purity': i_trans.get((u, v), 0) / ic if ic > 0 else 0.0,
                'prefix_purity': p_trans_strict.get((pu, pv), 0) / pc if pc > 0 else 0.0,
                'delta': delta
            })

    return pd.DataFrame(gen_rows), pd.DataFrame(mem_rows)


def report_generalization(df_gen, depths, n_bins=5):
    print("\n  --- Item Generalization: Performance vs Token Memorization Support Cn ---")
    
    for k in sorted(depths):
        df_k = df_gen[df_gen['k'] == k].copy()
        if df_k.empty: continue

        df_k['bin'] = '0'
        mask_pos = df_k['count'] > 0
        
        if mask_pos.any():
            # get intervals and calculate the integer midpoint for the label
            intervals = pd.qcut(df_k.loc[mask_pos, 'count'], q=n_bins, duplicates='drop')
            df_k.loc[mask_pos, 'bin'] = [str(int((iv.left + iv.right) / 2)) for iv in intervals]

        agg = df_k.groupby('bin').agg(
            N=('count', 'size'), 
            TIGER=('tiger', 'mean'), 
            SASRec=('sasrec', 'mean')
        )
        agg['Δ'] = agg['TIGER'] - agg['SASRec']
        
        # sort indicies
        def _sort_key(idx):
            return [-1.0 if str(x) == '0' else float(x) for x in idx]
        
        agg = agg.sort_index(key=_sort_key)
        
        print(f"\n  [n={k}] ({len(df_k)} instances)")
        print(f"    {'Cn':>8}  {'N':>6}  {'TIGER':>8}  {'SASRec':>8}  {'Δ':>8}")
        print(f"    {'─'*8}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
        for cn, r in agg.iterrows():
            print(f"    {cn:>8}  {int(r['N']):>6}  {r['TIGER']:>8.4f}  {r['SASRec']:>8.4f}  {r['Δ']:>+8.4f}")


def report_memorization(df_mem, n_bins=4):
    print("\n  --- Item Memorization: Δ NDCG by φ (item) × ψ (prefix) ---")
    
    df_mem['ψ_bin'] = pd.qcut(df_mem['prefix_purity'], q=n_bins, duplicates='drop')
    df_mem['φ_bin'] = pd.qcut(df_mem['item_purity'], q=n_bins, duplicates='drop')

    pv_mean = df_mem.pivot_table(index='φ_bin', columns='ψ_bin', values='delta', aggfunc='mean')
    pv_count = df_mem.pivot_table(index='φ_bin', columns='ψ_bin', values='delta', aggfunc='size')

    pv_mean = pv_mean.sort_index(ascending=False)
    pv_count = pv_count.sort_index(ascending=False)

    x_labels = [f"[{iv.left:.2f},{iv.right:.2f}]" for iv in pv_mean.columns]
    y_labels = [f"[{iv.left:.2f},{iv.right:.2f}]" for iv in pv_mean.index]

    nx, ny = len(x_labels), len(y_labels)
    col_w = max(len(l) for l in x_labels) + 2
    row_w = max(len(l) for l in y_labels) + 2

    print(f"\n  ψ = prefix transition prob (cols) × φ = item transition prob (rows)")
    print(f"  Δ NDCG = TIGER - SASRec")
    print(f"  {len(df_mem)} memorization instances, {nx}×{ny} grid\n")

    header = " " * (row_w + 2) + "".join(f"{l:>{col_w}}" for l in x_labels)
    print(f"  {'ψ →':>{row_w}}  {header.strip()}")
    print(f"  {'φ ↓':>{row_w}}  {'─' * (col_w * nx)}")

    for yi in range(ny):
        cells = []
        for xi in range(nx):
            v = pv_mean.iloc[yi, xi]
            c = pv_count.iloc[yi, xi]
            if pd.isna(v) or pd.isna(c) or c == 0:
                cells.append(f"{'—':>{col_w}}")
            else:
                cells.append(f"{v:>+{col_w-1}.3f} " if col_w > 7 else f"{v:>+.3f}")
        print(f"  {y_labels[yi]:>{row_w}}  {''.join(cells)}")

    print(f"  {' ' * row_w}  {'─' * (col_w * nx)}")
    print(f"  {'N':>{row_w}}  " + "".join(f"{int(pv_count.iloc[:, xi].sum()):>{col_w}}" for xi in range(nx)))

    print(f"\n  Overall Δ NDCG: {df_mem['delta'].mean():+.4f} ({len(df_mem)} instances)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--category", default=None)
    p.add_argument("--version", default=None)
    p.add_argument("--sem_ids_path", required=True)
    p.add_argument("--sasrec_infer_path", required=True)
    p.add_argument("--tiger_infer_path", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    ds_id = f"{args.dataset}-{args.version or args.category or ''}".rstrip('-')
    out_prefix = os.path.join(args.output_dir, f"token_mem_{ds_id}_{args.split}")
    
    with open(f"{out_prefix}_meta.json") as f:
        meta = json.load(f)
    df_clean = pd.read_csv(f"{out_prefix}_df_clean.csv")

    tiger_pipeline = Pipeline(
        model_name='TIGER', dataset_name=args.dataset,
        config_dict={'logging': False, 'sem_ids_path': args.sem_ids_path, 
                     'category': args.category, 'version': args.version}
    )
    
    # count transitions
    stats, get_prefix = compute_transition_stats(
        tiger_pipeline.split_datasets['train']['item_seq'], 
        tiger_pipeline.tokenizer, meta['DEPTHS'], meta['TARGET_K'], meta['K_WINDOW']
    )

    # conduct analysis
    df_gen, df_mem = build_analysis_dfs(
        df_clean, tiger_pipeline.split_datasets[args.split]['item_seq'], 
        stats, meta['DEPTHS'], meta['TARGET_K'], get_prefix
    )

    # save and report
    df_gen.to_csv(f"{out_prefix}_df_gen_clean.csv", index=False)
    df_mem.to_csv(f"{out_prefix}_df_mem_clean.csv", index=False)
    
    print(f"\n{'=' * 70}\n Performance Analysis — {ds_id}\n{'=' * 70}")
    report_generalization(df_gen, meta['DEPTHS'])
    report_memorization(df_mem)
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()