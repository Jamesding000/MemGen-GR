"""
Support coverage: how item generalization categories map to token memorization depth.
"""

import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mem_gen_categorizer import FineGrainedEvaluator
from token_mem_categorizer import PrefixGramMemorizationEvaluator
from genrec.pipeline import Pipeline


def get_item_case_labels(test_item_seqs, fine_grained_evaluator):
    """Return mapping: sample_idx → set of item-level case labels."""
    item_case_labels = {}
    for idx, item_seq in enumerate(test_item_seqs):
        item_case_labels[idx] = fine_grained_evaluator.get_case_labels(item_seq)
    return item_case_labels


def build_prefix_evaluators(train_item_seqs, tokenizer, prefix_lengths, max_hop):
    """Create one prefix-level memorization evaluator per prefix length."""
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
    """Return mapping: sample_idx → {prefix_len → set of token-level labels}."""
    token_case_labels = {}
    for idx, item_seq in tqdm(
        enumerate(test_item_seqs),
        total=len(test_item_seqs),
        desc="Token labels",
    ):
        token_case_labels[idx] = {}
        for plen in prefix_lengths:
            token_case_labels[idx][plen] = prefix_evaluators[plen].get_case_labels(item_seq)
    return token_case_labels


def get_token_category(token_labels_dict, prefix_lengths):
    """Map token-level labels to a single category like '3-gram' or 'unseen'."""
    for pl in sorted(prefix_lengths, reverse=True):
        labels = token_labels_dict.get(pl, set())
        if labels and 'unseen' not in labels:
            return f'{pl}-gram'
    return 'unseen'


def is_item_generalization(item_labels):
    """Return True if the item is a generalization case (not memorization)."""
    if 'memorization' in item_labels:
        return False
    sub_cats = ['substitutability', 'symmetry', 'transitivity', '2nd-symmetry', 'uncategorized']
    return any(label.split('_')[0] in sub_cats for label in item_labels)


def get_token_mem_ratio_by_subcat(item_case_labels, token_case_labels, prefix_lengths):
    """Return (pivot_pct, totals) for item subcategory × token memorization depth.

    Rows:  item-level generalization subcategories (e.g., symmetry, transitivity).
    Cols:  token memorization categories (e.g., 1-gram, 2-gram, unseen).
    Cells: percentage of samples in that item subcategory mapping to that token category.
    """
    records = []
    for idx, labels in item_case_labels.items():
        if not is_item_generalization(labels):
            continue
        token_cat = get_token_category(token_case_labels[idx], prefix_lengths)
        for label in labels:
            if label in ('memorization', 'generalization'):
                continue
            base = label.split('_')[0]
            records.append((base, token_cat))
    
    df = pd.DataFrame(records, columns=['item_subcat', 'token_cat'])
    counts = (
        df.pivot_table(
            index='item_subcat',
            columns='token_cat',
            aggfunc='size',
            fill_value=0,
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    totals = counts.sum(axis=1)
    pivot_pct = counts.div(totals, axis=0) * 100.0
    return pivot_pct, totals


def print_summary(pivot_pct, totals, dataset_id):
    """Print the conversion pivot in a compact text table."""
    if pivot_pct.empty:
        print("\nNo generalization cases found; nothing to report.\n")
        return

    display = pivot_pct.round(1)
    display.index = [f"{idx} (n={int(totals[idx])})" for idx in display.index]

    print(f"\n{'=' * 70}")
    print(f"  Item → Token conversion (generalization only) — {dataset_id}")
    print(f"{'=' * 70}")
    print(display.to_string(float_format=lambda v: f"{v:5.1f}"))
    print("\nValues are row-wise percentages for each item category.\n")


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
    args = p.parse_args()

    dataset_id = args.dataset
    if args.version:
        dataset_id = f"{args.dataset}-{args.version}"
    elif args.category:
        dataset_id = f"{args.dataset}-{args.category}"

    # setup pipelines
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

    # compute labels
    print("Computing item-level labels...")
    fg_evaluator = FineGrainedEvaluator(train_item_seqs=train_item_seqs, max_hop=args.max_hop)
    item_case_labels = get_item_case_labels(test_item_seqs, fg_evaluator)

    print("Computing token-level labels...")
    prefix_evaluators = build_prefix_evaluators(
        train_item_seqs, tiger_tokenizer, prefix_lengths, args.max_hop)
    token_case_labels = get_token_case_labels(test_item_seqs, prefix_evaluators, prefix_lengths)

    # construct memorization ratio table
    pivot_pct, totals = get_token_mem_ratio_by_subcat(
        item_case_labels, token_case_labels, prefix_lengths
    )
    print_summary(pivot_pct, totals, dataset_id)


if __name__ == "__main__":
    main()
