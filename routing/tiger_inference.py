import argparse
import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import logging
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from genrec.utils import get_config, init_seed, get_dataset, get_tokenizer
from genrec.models.TIGER.model import TIGER
from genrec.evaluator import Evaluator


class DummyAccelerator:
    def __init__(self):
        self.is_main_process = True
    @contextmanager
    def main_process_first(self):
        yield

def extract_category_from_checkpoint(checkpoint_path):
    """Extract category/version information from checkpoint filename."""
    basename = os.path.basename(checkpoint_path)
    match = re.search(r'--category=([^-]+)', basename)
    if match:
        return match.group(1)
    match = re.search(r'category_([^.]+)', basename)
    if match:
        return match.group(1)
    match = re.search(r'version_([^.]+)', basename)
    if match:
        return match.group(1)

    return None

def get_dataset_identifier(dataset_name, category=None, version=None):
    """Generate dataset identifier for output file naming."""
    if 'Yelp' in dataset_name and version:
        return f"{dataset_name}-{version}"
    elif category:
        return f"{dataset_name}-{category}"
    else:
        return dataset_name

def main():
    parser = argparse.ArgumentParser(description="Inference for a single TIGER model")
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--config_file', type=str, default='genrec/models/TIGER/config.yaml')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--category', type=str, default=None,
                        help='Dataset category. If not specified, will extract from checkpoint filename.')
    parser.add_argument('--version', type=str, default=None,
                        help='Dataset version (for Yelp dataset). If not specified, will extract from checkpoint filename.')
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--d_ff', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--num_decoder_layers', type=int, required=True)
    parser.add_argument('--num_heads', type=int, required=True)
    parser.add_argument('--d_kv', type=int, required=True)
    parser.add_argument('--sem_ids_path', type=str, default=None)
    parser.add_argument('--n_predictions', type=int, default=10,
                        help='Number of top predictions (beams) to save per sample (default: 10)')

    args = parser.parse_args()

    extracted_info = extract_category_from_checkpoint(args.model_ckpt)
    if 'Yelp' in args.dataset:
        if args.version is None and extracted_info:
            args.version = extracted_info
    else:
        if args.category is None and extracted_info:
            args.category = extracted_info

    config = get_config('TIGER', args.dataset, args.config_file, None)
    if args.category is not None:
        config['category'] = args.category
    if args.version is not None:
        config['version'] = args.version
    if args.sem_ids_path:
        config['sem_ids_path'] = args.sem_ids_path

    init_seed(config['rand_seed'], config['reproducibility'])
    config['accelerator'] = DummyAccelerator()

    log_dir = os.path.join(config['log_dir'], config['dataset'], config['model'])
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"inference.log")
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()], force=True)

    config['metrics'] = ['ndcg', 'recall']

    raw_dataset = get_dataset(args.dataset)(config)
    split_datasets = raw_dataset.split()
    tokenizer = get_tokenizer('TIGER')(config, raw_dataset)
    tokenized_datasets = tokenizer.tokenize(split_datasets)
    evaluator = Evaluator(config, tokenizer)
    
    semantic_to_item = {}
    for item, tokens in tokenizer.item2tokens.items():
        semantic_to_item[tokens] = raw_dataset.item2id[item]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_config_keys = ['dataset', 'log_dir', 'model', 'eval_batch_size', 'rand_seed', 'reproducibility', 'accelerator',
                        'topk', 'num_beams', 'confidence_top_k', 'dropout_rate', 'activation_function', 'feed_forward_proj']
    model_config = {key: config[key] for key in base_config_keys if key in config}
    model_config.update({
        'd_model': args.d_model, 'd_ff': args.d_ff, 'num_layers': args.num_layers,
        'num_decoder_layers': args.num_decoder_layers, 'num_heads': args.num_heads, 'd_kv': args.d_kv
    })

    model = TIGER(model_config, raw_dataset, tokenizer).to(device)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.eval()

    dataloader = DataLoader(
        tokenized_datasets[args.split],
        batch_size=config['eval_batch_size'],
        shuffle=False,
        collate_fn=tokenizer.collate_fn['test']
    )
    
    target_beam_size = config.get('num_beams', 10)
    n_predictions = args.n_predictions
    assert target_beam_size >= n_predictions, "num_beams must be >= n_predictions"
    
    all_results_list = []
    sample_id_counter = 0
    total_invalid_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluation on {args.split}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            generate_output = model.generate(batch, n_return_sequences=target_beam_size, return_scores=True)
            preds = generate_output['preds']
            labels_raw = batch['labels']
            preds_for_eval = preds[:, :evaluator.maxk] 

            pos_index = evaluator.calculate_pos_index(preds_for_eval, labels_raw)
            results_per_batch = {}
            all_metrics_to_calc = set(config.get('metrics', []))

            for k in config['topk']:
                if k <= evaluator.maxk:
                    if 'recall' in all_metrics_to_calc:
                        results_per_batch[f'recall@{k}'] = evaluator.recall_at_k(pos_index, k)
                    if 'ndcg' in all_metrics_to_calc:
                        results_per_batch[f'ndcg@{k}'] = evaluator.ndcg_at_k(pos_index, k)

            num_beams_eval = preds.shape[1]
            beam_scores = generate_output['scores'].cpu().numpy()
            
            for key, val in results_per_batch.items():
                arr = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
                results_per_batch[key] = arr.reshape(-1)

            num_records_in_batch = batch['input_ids'].shape[0] * num_beams_eval
            preds_np = preds.cpu().numpy()
            
            batch_pred_items = []
            for b in range(preds_np.shape[0]):
                for beam in range(preds_np.shape[1]):
                    semantic_id = tuple(preds_np[b, beam, :].tolist())
                    item_id = semantic_to_item.get(semantic_id, -1)
                    if item_id == -1:
                        total_invalid_count += 1
                    batch_pred_items.append(item_id)
            
            for i in range(num_records_in_batch):
                sample_idx = i // num_beams_eval
                beam_rank = i % num_beams_eval
                if beam_rank >= n_predictions:
                    continue

                record = {
                    'sample_id': sample_id_counter + sample_idx, 
                    'beam_rank': beam_rank,
                    'pred_item': batch_pred_items[i],
                }
                
                if sample_idx < len(labels_raw):
                    target_semantic = labels_raw[sample_idx].cpu().numpy()
                    target_semantic_tuple = tuple(target_semantic[:tokenizer.n_digit].tolist())
                    target_item = semantic_to_item.get(target_semantic_tuple, -1)
                    record['target_item'] = target_item
                
                if beam_scores is not None:
                    record['beam_score'] = beam_scores[sample_idx, beam_rank]
                
                for key, flat_array in results_per_batch.items():
                    if len(flat_array) == batch['input_ids'].shape[0]:
                        record[key] = flat_array[sample_idx]
                    elif len(flat_array) == num_records_in_batch:
                        record[key] = flat_array[i]
                    else:
                        record[key] = float('nan')

                all_results_list.append(record)
            
            sample_id_counter += batch['input_ids'].shape[0]

    df_results = pd.DataFrame(all_results_list)
    dataset_identifier = get_dataset_identifier(
        args.dataset,
        category=config.get('category'),
        version=config.get('version')
    )
    output_path = f"Confidence/TIGER/results/tiger_predictions_with_scores_{args.split}_{dataset_identifier}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)

    df_top_beam = df_results[df_results['beam_rank'] == 0]
    if not df_top_beam.empty:
        avg_results = df_top_beam[[col for col in df_results.columns if col not in ['sample_id', 'beam_rank']]].mean().sort_index()
        print(f"Results saved to: {output_path}")
        if 'ndcg@10' in avg_results:
            print(f"NDCG@10: {avg_results['ndcg@10']:.4f}")
        if 'recall@10' in avg_results:
            print(f"Recall@10: {avg_results['recall@10']:.4f}")

if __name__ == '__main__':
    main()