"""
Section 4.2 From Item Generalization to Token Memorization
"""

import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mem_gen_categorizer import FineGrainedEvaluator
from token_mem_categorizer import PrefixGramMemorizationEvaluator
from genrec.pipeline import Pipeline

# Label helpers
def get_item_case_labels(test_item_seqs, fine_grained_evaluator):
    item_case_labels = {}
    for idx, item_seq in enumerate(test_item_seqs):
        item_case_labels[idx] = fine_grained_evaluator.get_case_labels(item_seq)
    return item_case_labels


def build_prefix_evaluators(train_item_seqs, tokenizer, prefix_lengths, max_hop):
    evaluators = {}
    for plen in prefix_lengths:
        evaluators[plen] = PrefixGramMemorizationEvaluator(
            train_item_seqs=train_item_seqs,
            tokenizer=tokenizer,
            prefix_length=plen,
            max_hop=max_hop,
        )
    return evaluators


def get_token_case_labels(test_item_seqs, prefix_evaluators, prefix_lengths):
    token_case_labels = {}
    for idx, item_seq in tqdm(enumerate(test_item_seqs), total=len(test_item_seqs),
                              desc="Token labels"):
        token_case_labels[idx] = {}
        for plen in prefix_lengths:
            token_case_labels[idx][plen] = prefix_evaluators[plen].get_case_labels(item_seq)
    return token_case_labels


def get_token_category(token_labels_dict, prefix_lengths):
    for pl in sorted(prefix_lengths, reverse=True):
        labels = token_labels_dict.get(pl, set())
        if labels and 'unseen' not in labels:
            return f'{pl}-gram'
    return 'unseen'


def is_item_generalization(item_labels):
    if 'memorization' in item_labels:
        return False
    sub_cats = ['substitutability', 'symmetry', 'transitivity', '2nd-symmetry', 'uncategorized']
    return any(label.split('_')[0] in sub_cats for label in item_labels)


def compute_conversion_rates(item_case_labels, token_case_labels, prefix_lengths):
    gen_indices = [idx for idx in item_case_labels
                   if is_item_generalization(item_case_labels[idx])]
    n_gen = len(gen_indices)
    if n_gen == 0:
        return {plen: 0.0 for plen in prefix_lengths}
    rates = {}
    for plen in prefix_lengths:
        token_cat = f'{plen}-gram'
        n_convert = sum(
            1 for idx in gen_indices
            if get_token_category(token_case_labels[idx], prefix_lengths) == token_cat
        )
        rates[plen] = 100 * n_convert / n_gen
    return rates

# Data building
def build_df_clean(tiger_infer_path, sasrec_infer_path,
                   item_case_labels, token_case_labels, prefix_lengths):
    df_tiger = pd.read_csv(tiger_infer_path)
    df_sasrec = pd.read_csv(sasrec_infer_path)

    rank_col = 'rank_id' if 'rank_id' in df_tiger.columns else 'rank'
    df_tiger_top1 = df_tiger[df_tiger[rank_col] == 1].copy()
    df_sasrec_top1 = df_sasrec[df_sasrec[rank_col] == 1].copy()

    df_tiger_top1 = df_tiger_top1.rename(columns={'ndcg@10': 'ndcg@10_tiger'})
    df_sasrec_top1 = df_sasrec_top1.rename(columns={'ndcg@10': 'ndcg@10_sasrec'})

    df_clean = df_tiger_top1.merge(
        df_sasrec_top1[['sample_id', 'ndcg@10_sasrec']],
        on='sample_id', how='inner',
    )

    df_clean['is_item_generalization'] = df_clean['sample_id'].map(
        lambda sid: is_item_generalization(item_case_labels[sid])
    )
    df_clean['token_cat'] = df_clean['sample_id'].map(
        lambda sid: get_token_category(token_case_labels[sid], prefix_lengths)
    )
    df_clean['token_depth'] = df_clean['token_cat'].apply(
        lambda x: 0 if x == 'unseen' else int(x.split('-')[0])
    )
    return df_clean


def build_conversion_df(n_test, item_case_labels, token_case_labels, prefix_lengths):
    sub_cats_rows = ['symmetry', 'transitivity', '2nd-symmetry', 'uncategorized']
    records = []
    for idx in range(n_test):
        token_cat = get_token_category(token_case_labels[idx], prefix_lengths)
        for label in item_case_labels[idx]:
            if label in ('memorization', 'generalization'):
                continue
            base = label.split('_')[0]
            if base in sub_cats_rows:
                records.append({'item_subcat': base, 'token_cat': token_cat, 'idx': idx})
    return pd.DataFrame(records)


def build_category_dicts(n_test, item_case_labels, token_case_labels,
                         prefix_lengths, fine_grained_evaluator):
    main_categories = ['memorization', 'generalization', 'uncategorized']
    sub_categories = ['memorization', 'substitutability', 'symmetry',
                      'transitivity', '2nd-symmetry', 'uncategorized']
    sub_categories_with_hop = fine_grained_evaluator.ordered_keys
    token_mem_categories = [f'{pl}-gram' for pl in prefix_lengths] + ['unseen']

    main_dict = {lab: set() for lab in main_categories}
    sub_dict = {lab: set() for lab in sub_categories}
    sub_hop_dict = {lab: set() for lab in sub_categories_with_hop}
    token_dict = {lab: set() for lab in token_mem_categories}

    for idx in range(n_test):
        labels = item_case_labels[idx]
        token_labs = token_case_labels[idx]

        for label in labels:
            if label == 'generalization':
                continue
            sub_hop_dict[label].add(idx)
            sub_dict[label.split('_')[0]].add(idx)

        if 'memorization' in labels:
            main_dict['memorization'].add(idx)
        elif 'uncategorized' in labels:
            main_dict['uncategorized'].add(idx)
        else:
            main_dict['generalization'].add(idx)

        is_token_mem = False
        for pl in prefix_lengths[::-1]:
            if 'unseen' not in token_labs[pl]:
                token_dict[f'{pl}-gram'].add(idx)
                is_token_mem = True
                break
        if not is_token_mem:
            token_dict['unseen'].add(idx)

    return {
        'main_categories': main_categories,
        'sub_categories': sub_categories,
        'sub_categories_with_hop': sub_categories_with_hop,
        'token_mem_categories': token_mem_categories,
        'main_categories_dict': {k: sorted(list(v)) for k, v in main_dict.items()},
        'sub_categories_dict': {k: sorted(list(v)) for k, v in sub_dict.items()},
        'sub_categories_with_hop_dict': {k: sorted(list(v)) for k, v in sub_hop_dict.items()},
        'token_mem_categories_dict': {k: sorted(list(v)) for k, v in token_dict.items()},
    }

# Save / report
def save_results(out_prefix, item_case_labels, token_case_labels,
                 df_clean, conversion_df, category_dicts, meta):
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)

    with open(f"{out_prefix}_item_case_labels.json", 'w') as f:
        json.dump({str(k): sorted(list(v)) for k, v in item_case_labels.items()}, f)

    with open(f"{out_prefix}_token_case_labels.json", 'w') as f:
        tcl = {
            str(k): {str(pk): sorted(list(pv)) for pk, pv in v.items()}
            for k, v in token_case_labels.items()
        }
        json.dump(tcl, f)

    df_clean[['sample_id', 'ndcg@10_tiger', 'ndcg@10_sasrec',
              'is_item_generalization', 'token_cat', 'token_depth']
    ].to_csv(f"{out_prefix}_df_clean.csv", index=False)

    conversion_df.to_csv(f"{out_prefix}_conversion_df.csv", index=False)

    with open(f"{out_prefix}_category_dicts.json", 'w') as f:
        json.dump(category_dicts, f)

    with open(f"{out_prefix}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)


def print_summary(df_clean, conversion_df, prefix_lengths,
                  item_case_labels, token_case_labels, dataset_id):
    n_gen = int(df_clean['is_item_generalization'].sum())
    n_mem = int((~df_clean['is_item_generalization']).sum())
    rates = compute_conversion_rates(item_case_labels, token_case_labels, prefix_lengths)

    print(f"\n{'=' * 60}")
    print(f"  Support Coverage — {dataset_id}")
    print(f"{'=' * 60}")
    print(f"  Total samples:  {len(df_clean)}")
    print(f"  Generalization: {n_gen}")
    print(f"  Memorization:   {n_mem}")
    print(f"\n  {'Prefix':>8}  {'Coverage %':>12}")
    print(f"  {'-' * 8}  {'-' * 12}")
    for plen in prefix_lengths:
        print(f"  {plen}-gram{' ' * (4 - len(str(plen)))}  {rates[plen]:>11.1f}%")
    print(f"\n  Token category distribution:")
    for cat, count in df_clean['token_cat'].value_counts().items():
        print(f"    {cat}: {count} ({100 * count / len(df_clean):.1f}%)")
    print(f"{'=' * 60}\n")

def main():
    p = argparse.ArgumentParser(description="Support coverage analysis")
    p.add_argument("--dataset", required=True)
    p.add_argument("--category", default=None)
    p.add_argument("--version", default=None)
    p.add_argument("--sem_ids_path", required=True)
    p.add_argument("--tiger_infer_path", required=True)
    p.add_argument("--sasrec_infer_path", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max_hop", type=int, default=4)
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    dataset_id = args.dataset
    if args.version:
        dataset_id = f"{args.dataset}-{args.version}"
    elif args.category:
        dataset_id = f"{args.dataset}-{args.category}"

    os.makedirs(args.output_dir, exist_ok=True)
    out_prefix = os.path.join(args.output_dir, f"token_mem_{dataset_id}_{args.split}")

    # --- Build pipelines ---
    config_tiger = {'logging': False, 'sem_ids_path': args.sem_ids_path}
    config_sasrec = {'logging': False}
    if args.category:
        config_tiger['category'] = args.category
        config_sasrec['category'] = args.category
    if args.version:
        config_tiger['version'] = args.version
        config_sasrec['version'] = args.version

    print("Loading TIGER pipeline (for tokenizer)...")
    tiger_pipeline = Pipeline(model_name='TIGER', dataset_name=args.dataset,
                              config_dict=config_tiger)
    tiger_tokenizer = tiger_pipeline.tokenizer

    print("Loading SASRec pipeline (for dataset splits)...")
    sasrec_pipeline = Pipeline(model_name='SASRec', dataset_name=args.dataset,
                               config_dict=config_sasrec)

    prefix_lengths = list(range(1, tiger_tokenizer.n_digit + 1))
    sem_prefix_lengths = list(range(1, tiger_tokenizer.n_digit))

    train_item_seqs = sasrec_pipeline.split_datasets['train']['item_seq']
    test_item_seqs = sasrec_pipeline.split_datasets[args.split]['item_seq']
    n_test = len(test_item_seqs)

    # --- Compute labels ---
    print("Computing item-level labels...")
    fg_evaluator = FineGrainedEvaluator(train_item_seqs=train_item_seqs, max_hop=args.max_hop)
    item_case_labels = get_item_case_labels(test_item_seqs, fg_evaluator)

    print("Computing token-level labels...")
    prefix_evaluators = build_prefix_evaluators(
        train_item_seqs, tiger_tokenizer, prefix_lengths, args.max_hop)
    token_case_labels = get_token_case_labels(test_item_seqs, prefix_evaluators, prefix_lengths)

    # --- Build dataframes ---
    print("Building df_clean...")
    df_clean = build_df_clean(args.tiger_infer_path, args.sasrec_infer_path,
                              item_case_labels, token_case_labels, prefix_lengths)

    print("Building conversion_df...")
    conversion_df = build_conversion_df(n_test, item_case_labels, token_case_labels, prefix_lengths)

    print("Building category dicts...")
    category_dicts = build_category_dicts(
        n_test, item_case_labels, token_case_labels, prefix_lengths, fg_evaluator)

    # --- Metadata ---
    max_sem_depth = max(sem_prefix_lengths) if sem_prefix_lengths else max(prefix_lengths)
    l_analyze = min(3, max_sem_depth)
    meta = {
        'DATASET_ID': dataset_id,
        'DATASET_NAME': args.dataset,
        'CATEGORY': args.category or '',
        'VERSION': args.version or '',
        'SPLIT': args.split,
        'MAX_HOP': args.max_hop,
        'PREFIX_LENGTHS': prefix_lengths,
        'SEM_PREFIX_LENGTHS': sem_prefix_lengths,
        'DEPTHS': list(range(1, l_analyze + 1)),
        'L_ANALYZE': l_analyze,
        'K_WINDOW': 4,
        'TARGET_K': max(1, min(2, max_sem_depth)),
    }

    # --- Save & report ---
    save_results(out_prefix, item_case_labels, token_case_labels,
                 df_clean, conversion_df, category_dicts, meta)
    print(f"All results saved to {out_prefix}_*")

    print_summary(df_clean, conversion_df, prefix_lengths,
                  item_case_labels, token_case_labels, dataset_id)


if __name__ == "__main__":
    main()
