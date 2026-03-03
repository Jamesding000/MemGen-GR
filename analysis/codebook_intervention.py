"""
Section 4.4: Codebook Intervention Analysis
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mem_gen_categorizer import FineGrainedEvaluator
from token_mem_categorizer import PrefixGramMemorizationEvaluator
from genrec.pipeline import Pipeline

# label helpers
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
    return 'memorization' not in item_labels


def load_experiments_from_sweep(sweep_path, dataset, category, results_dir):
    """Load experiment configs from sweep.yaml and derive all paths."""
    with open(sweep_path) as f:
        sweep = yaml.safe_load(f)

    experiments = {}
    for entry in sweep['parameters']['experiment_tuple']['values']:
        cb_size, n_cb, budget = entry.split(':')
        cb_size, n_cb, budget = int(cb_size), int(n_cb), int(budget)
        config_name = f"{cb_size}x{n_cb}"

        sizes = ','.join([str(cb_size)] * n_cb)
        sub = f"{dataset}/{category}" if category else dataset
        sem_ids_path = f"cache/{sub}/processed/sentence-t5-base_{sizes}.sem_ids"
        os.makedirs(os.path.join(results_dir, sweep['project']), exist_ok=True)
        result_path = os.path.join(results_dir, f"{sweep['project']}/codebook_{config_name}.csv")

        experiments[config_name] = {
            'result_path': result_path,
            'sem_ids_path': sem_ids_path,
            'budget': budget,
        }
    return experiments


def parse_sid_config(config_name):
    """Parse 'SIZExDEPTH' (e.g. '256x4') -> (codebook_size, n_codebooks)."""
    parts = config_name.split('x')
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    return None, None


def process_eval_results(experiments):
    all_val_rows = []
    all_test_rows = []

    for sid, exp in experiments.items():
        result_path = exp['result_path']
        budget_epoch = exp.get('budget')

        assert os.path.exists(result_path), f"Result file not found: {result_path}"
        df = pd.read_csv(result_path)
        cb_size, n_cb = parse_sid_config(sid)

        # extract validation dynamics
        val_mask = df['split'] == 'val'
        if budget_epoch is not None:
            val_mask = val_mask & (df['epoch'] <= budget_epoch)
        val_df = df.loc[val_mask, ['epoch', 'FG/memorization', 'FG/generalization']].copy()
        if len(val_df) > 0:
            val_df['sid'] = sid
            val_df['codebook_size'] = cb_size
            val_df['sid_length'] = n_cb
            all_val_rows.append(val_df)

        # extract test summary
        assert df[df['split'] == 'test'].shape[0] > 0, "No test data found."
        
        test_epochs = df[df['split'] == 'test']['epoch']
        test_mask = (df['split'] == 'test') & (df['epoch'] == budget_epoch or test_epochs.max())
        test_df = df.loc[test_mask, ['epoch', 'FG/memorization', 'FG/generalization']].copy()
        
        if len(test_df) > 0:
            test_df['sid'] = sid
            test_df['codebook_size'] = cb_size
            test_df['sid_length'] = n_cb
            all_test_rows.append(test_df)
        else:
            print(f"Warning: no test data at epoch {budget_epoch} for {sid}")

    val_dynamics = pd.concat(all_val_rows, ignore_index=True) if all_val_rows else pd.DataFrame()
    test_summary = pd.concat(all_test_rows, ignore_index=True) if all_test_rows else pd.DataFrame()
    return val_dynamics, test_summary


def get_token_mem_ratio_by_prefix_length(item_case_labels, token_case_labels, prefix_lengths):
    max_k = max(prefix_lengths)
    gen_indices = [idx for idx in item_case_labels
                   if is_item_generalization(item_case_labels[idx])]
    n_gen = len(gen_indices)
    if n_gen == 0:
        return {k: 0.0 for k in range(max_k + 1)}

    rates = {k: 0.0 for k in range(max_k + 1)}
    for idx in gen_indices:
        token_cat = get_token_category(token_case_labels[idx], prefix_lengths)
        if token_cat == 'unseen':
            rates[0] += 1
        else:
            k = int(token_cat.split('-')[0])
            if k in rates:
                rates[k] += 1

    for k in rates:
        rates[k] = 100.0 * rates[k] / n_gen
    return rates


def resolve_sem_ids_path(base_path):
    if os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    alt = base + '_faiss' + ext
    return alt if os.path.exists(alt) else base_path


def build_conversion_table(experiments, dataset_name, category, version,
                           item_case_labels, test_item_seqs, max_hop):
    prefix_cols = list(range(0, 7))
    rows = []

    for config_name, exp in tqdm(experiments.items(), desc="SID conversion rates"):
        sem_ids_path = exp['sem_ids_path']

        resolved = resolve_sem_ids_path(sem_ids_path)
        assert os.path.exists(resolved), f"Path not found: {resolved}"

        cb_size, n_cb = parse_sid_config(config_name)
        config_dict = {
            'logging': False,
            'sem_ids_path': resolved,
            'rq_n_codebooks': n_cb,
            'rq_codebook_size': cb_size,
        }
        if category:
            config_dict['category'] = category
        if version:
            config_dict['version'] = version

        pipe = Pipeline(model_name='TIGER', dataset_name=dataset_name,
                        config_dict=config_dict)
        tok = pipe.tokenizer
        prefix_lengths = list(range(1, n_cb + 2))

        train_seqs = pipe.split_datasets['train']['item_seq']
        peval = build_prefix_evaluators(train_seqs, tok, prefix_lengths, max_hop)
        tlabels = get_token_case_labels(test_item_seqs, peval, prefix_lengths)

        rates = get_token_mem_ratio_by_prefix_length(item_case_labels, tlabels, prefix_lengths)
        row = {'config': config_name}
        for k in prefix_cols:
            row[k] = rates.get(k, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index('config')[prefix_cols]
    return df


def print_report(val_dynamics, test_summary, conversion_df,
                 dataset_id):
    print(f"\n{'=' * 70}")
    print(f"  Codebook Intervention Analysis — {dataset_id}")
    print(f"{'=' * 70}")

    # conversion rates
    print(f"\n  --- Support Coverage / Conversion Rates (%) ---")
    print(f"  Rows: SID configs | Cols: 0=unseen, 1..N = k-gram memorization")
    print(conversion_df.round(2).to_string())

    # validation dynamics
    print(f"\n  --- Validation Training Dynamics ---")
    dyn = val_dynamics.copy()
    tail = (dyn.sort_values('epoch')
            .groupby('sid').tail(3)
            .sort_values(['sid_length', 'codebook_size', 'epoch'],
                            ascending=[False, True, True]))
    display_cols = ['sid', 'epoch', 'FG/memorization', 'FG/generalization']
    avail = [c for c in display_cols if c in tail.columns]
    fmt = tail[avail].copy()
    for col in ['FG/memorization', 'FG/generalization']:
        fmt[col] = fmt[col].apply(lambda x: f"{x:.4f}")
    print(fmt.to_string(index=False))

    # test summary
    print(f"\n  --- Test Results (at budget epoch) ---")
    ts = test_summary.sort_values(['sid_length', 'codebook_size'],
                                    ascending=[False, True]).copy()
    display_cols = ['sid', 'epoch', 'FG/memorization', 'FG/generalization']
    avail = [c for c in display_cols if c in ts.columns]
    fmt = ts[avail].copy()
    for col in ['FG/memorization', 'FG/generalization']:
        fmt[col] = fmt[col].apply(lambda x: f"{x:.4f}")
    print(fmt.to_string(index=False))

    print(f"{'=' * 70}\n")

def main():
    p = argparse.ArgumentParser(description="Codebook intervention analysis")
    p.add_argument("--dataset", required=True)
    p.add_argument("--category", default=None)
    p.add_argument("--version", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--max_hop", type=int, default=4)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--results_dir", default="logs/fine_grained_results")
    p.add_argument("--sweep_config", required=True,
                   help="Path to sweep.yaml with experiment_tuple definitions")
    args = p.parse_args()

    dataset_id = args.dataset
    if args.version:
        dataset_id = f"{args.dataset}-{args.version}"
    elif args.category:
        dataset_id = f"{args.dataset}-{args.category}"

    os.makedirs(args.output_dir, exist_ok=True)

    experiments = load_experiments_from_sweep(
        args.sweep_config, args.dataset, args.category, args.results_dir)

    # --- 1. Process eval results ---
    print("Processing evaluation results...")
    val_dynamics, test_summary = process_eval_results(experiments)
    
    # --- 3. Conversion rates ---
    config_sasrec = {'logging': False}
    if args.category:
        config_sasrec['category'] = args.category
    if args.version:
        config_sasrec['version'] = args.version

    print("Loading SASRec pipeline for dataset...")
    sasrec_pipeline = Pipeline(model_name='SASRec', dataset_name=args.dataset,
                               config_dict=config_sasrec)
    train_item_seqs = sasrec_pipeline.split_datasets['train']['item_seq']
    test_item_seqs = sasrec_pipeline.split_datasets[args.split]['item_seq']

    print("Computing item-level labels...")
    fg_evaluator = FineGrainedEvaluator(train_item_seqs=train_item_seqs, max_hop=args.max_hop)
    item_case_labels = get_item_case_labels(test_item_seqs, fg_evaluator)

    print("Computing conversion rates per SID config...")
    conversion_df = build_conversion_table(
        experiments, args.dataset, args.category, args.version,
        item_case_labels, test_item_seqs, args.max_hop)

    # --- Save ---
    test_summary.to_csv(os.path.join(args.output_dir, 'codebook_test_summary.csv'), index=False)
    val_dynamics.to_csv(os.path.join(args.output_dir, 'codebook_val_dynamics.csv'), index=False)
    conversion_df.to_csv(os.path.join(args.output_dir, 'codebook_conversion_rates.csv'))

    print(f"All results saved to {args.output_dir}/codebook_*")

    # --- Report ---
    print_report(val_dynamics, test_summary, conversion_df, dataset_id)


if __name__ == "__main__":
    main()
