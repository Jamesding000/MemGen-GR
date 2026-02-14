import argparse
from collections import defaultdict
import csv
import os
import torch

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args
from genrec.fine_grained_evaluator import FineGrainedEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_dataset_id(config):
    if 'Yelp' in config['dataset'] and config['version']:
        return f"{config['dataset']}-{config['version']}"
    elif 'category' in config and config['category']:
        return f"{config['dataset']}-{config['category']}"
    else:
        return config['dataset']

class FineGrainedResultPipeline(Pipeline):
    def __init__(self, eval_set="test", log_file=None, save_inference=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_set = eval_set
        self.log_file = log_file
        self.save_inference = save_inference

        # Initialize fine-grained evaluator
        self.fine_grained_metric = self.config.get('fine_grained_metric', 'ndcg@10')
        self.fine_grained_evaluator = FineGrainedEvaluator(
            train_item_seqs=self.split_datasets['train']['item_seq'],
            max_hop=self.config.get('max_hop', 4),
        )
        
        # Setup inference results path
        dataset_id = get_dataset_id(self.config)
        if self.save_inference:
            os.makedirs('logs/inference_results', exist_ok=True)
            self.inference_file = f"logs/inference_results/{self.config['model']}_{dataset_id}_{self.eval_set}_inference_results.csv"

        # Build reverse mapping for (Tokens -> ItemID)
        assert hasattr(self.tokenizer, 'item2tokens'), "Tokenizer must have item2tokens attribute"
        self.tokens2item = {v if isinstance(v, int) else tuple(v): k for k, v in self.tokenizer.item2tokens.items()}
        
    def run(self):
        # Calculate metrics per test case
        eval_dataloader = DataLoader(
            self.tokenized_datasets[self.eval_set],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn[self.eval_set]
        )
        self.model, eval_dataloader = self.accelerator.prepare(self.model, eval_dataloader)

        # Initialize results storage
        test_cases = self.split_datasets[self.eval_set]['item_seq']
        case_labels = {}
        for idx, item_seq in enumerate(test_cases):
            labels = self.fine_grained_evaluator.get_case_labels(item_seq)
            case_labels[idx] = labels
        
        fine_grained_results = {label: [] for label in self.fine_grained_evaluator.ordered_keys}
        prediction_category_counts = {label: 0 for label in self.fine_grained_evaluator.ordered_keys}
        total_predictions_count = 0
        inference_data = [] if self.save_inference else None
        
        all_results = defaultdict(list)

        # Evaluate each test case
        self.model.eval()
        val_progress_bar = tqdm(
            eval_dataloader,
            total=len(eval_dataloader),
            desc=f"Eval - {self.eval_set}",
        )
        for batch_idx, batch in enumerate(val_progress_bar):
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

                if self.config['use_ddp']:
                    preds = self.model.module.generate(batch, n_return_sequences=10)
                    all_preds, all_labels, all_indices = self.accelerator.gather_for_metrics(
                        (preds, batch['labels'], batch['idx'])
                    )
                else:
                    all_preds = self.model.generate(batch, n_return_sequences=10)
                    all_labels = batch['labels']
                    all_indices = batch['idx']

                k = 10
                if all_preds.dim() == 2:
                    all_preds = all_preds.view(-1, k, all_preds.size(-1))

                batch_results = self.trainer.evaluator.calculate_metrics(all_preds, all_labels)
                for key, value in batch_results.items():
                    all_results[key].append(value)

                all_scores = batch_results[self.fine_grained_metric].cpu().tolist()
                all_indices = all_indices.cpu().tolist()
                all_preds = all_preds.cpu().tolist()
                all_labels = all_labels.cpu().tolist()

                for i, (case_idx, score) in enumerate(zip(all_indices, all_scores)):
                    if case_idx in case_labels:
                        categories = case_labels[case_idx]
                        for category in categories:
                            fine_grained_results[category].append(score)
                    
                    for pred_tokens in all_preds[i]:
                        pred_tuple = tuple(pred_tokens) if len(pred_tokens) > 1 else pred_tokens[0]
                        pred_item_id = self.tokens2item.get(pred_tuple, None)
                        if not pred_item_id:
                            pred_item_categories = {'novelty'}
                        else:
                            history_seq = test_cases[case_idx][:-1]
                            pred_item_categories = self.fine_grained_evaluator.get_case_labels(history_seq + [pred_item_id])
                        
                        for cat in pred_item_categories:
                            prediction_category_counts[cat] += 1
                            
                        total_predictions_count += 1

                    # Store detailed inference results
                    if self.save_inference:
                        pred = all_preds[i]
                        label = all_labels[i]
                        item_labels_str = str(list(case_labels[case_idx]))
            
                        # Create row for each prediction rank (up to 50)
                        max_k = min(50, len(pred))
                        for rank_id in range(max_k):
                            inference_data.append({
                                'sample_id': case_idx,
                                'rank_id': rank_id,
                                'prediction': pred[rank_id],
                                'label': label[:-1] if len(label) > 1 else label,  # a list for TIGER
                                'ndcg@5': batch_results['ndcg@5'][i].item(),
                                'ndcg@10': batch_results['ndcg@10'][i].item(),
                                'recall@5': batch_results['recall@5'][i].item(),
                                'recall@10': batch_results['recall@10'][i].item(),
                                'item_labels': item_labels_str,
                            })

        # Report aggregated metrics
        self.log('=' * 80)
        self.log(f'Aggregated Evaluation Metrics on {self.eval_set.upper()} set:')
        self.log('=' * 80)
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                if all_results[key]:
                    avg_score = torch.cat(all_results[key]).mean().item()
                    self.log(f'{key}: {avg_score:.6f}')
        self.log('')

        # Define output order to match paper table
        ordered_keys = self.fine_grained_evaluator.ordered_keys
        
        # Helper to print group stats
        total_cases = len(test_cases)
        for label_key in ordered_keys:
            if fine_grained_results[label_key]:
                avg_ndcg = sum(fine_grained_results[label_key]) / len(fine_grained_results[label_key])
                ratio = len(fine_grained_results[label_key]) / total_cases
                self.log(f'{label_key:<20}: NDCG@10 = {avg_ndcg:.4f} (n={len(fine_grained_results[label_key])}, ratio={ratio:.2%})')
            else:
                self.log(f'{label_key:<20}: NDCG@10 = N/A    (n=0, ratio=0.00%)')
        
        # fine_grained_results = self.fine_grained_evaluator.compute_fine_grained_metrics(
        #     batch_indices, 
        #     test_cases, 
        #     ndcg_10_scores,
        #     case_labels=case_labels
        # )
        
        # prediction_ratio = self.fine_grained_evaluator.compute_prediction_distribution(
        #     batch_indices, 
        #     test_cases, 
        #     preds, 
        #     self.tokens2item
        # )
        
        # Helper function to report prediction distribution
        self.log('=' * 80)
        self.log('Prediction Category Distribution (Model Behavior):')
        self.log('=' * 80)
        prediction_ratio = {}
        for label_key, count in prediction_category_counts.items():
            prediction_ratio[label_key] = count / total_predictions_count
            self.log(f'{label_key:<20}: Count = {count} (Ratio = {count / total_predictions_count:.2%})')
        self.log('')
        
        if self.log_file:
            self._write_to_csv(fine_grained_results, prediction_ratio)
        
        if self.save_inference and inference_data:
            self._write_inference_results(inference_data)
            self.log(f'Saved {len(inference_data)} inference results to {self.inference_file}')

        self.trainer.end()

    # def _report_prediction_distribution(self, prediction_ratio):
    #     """Report the distribution of categories in the model's own predictions."""
    #     self.log('=' * 80)
    #     self.log('Prediction Category Distribution (Model Behavior):')
    #     self.log('=' * 80)
    #     for label_key, ratio in prediction_ratio.items():
    #         self.log(f'{label_key:<20}: Ratio = {ratio:.2%}')
    #     self.log('')

    def _write_to_csv(self, fine_grained_results, prediction_ratio):
        """Write fine-grained results to CSV file aligned with paper columns."""
        total_cases = len(self.split_datasets[self.eval_set]["item_seq"])

        # Header correctly includes generalization and novelty at the end
        ordered_keys = self.fine_grained_evaluator.ordered_keys
        header = ["Category"] + ordered_keys

        # Check for header
        file_exists = os.path.exists(self.log_file)
        has_header = False
        if file_exists:
            with open(self.log_file, "r") as f:
                first_line = f.readline().strip()
                if first_line.split("\t") == header:
                    has_header = True

        # Prepare Data Rows
        dataset_id = get_dataset_id(self.config)
        category = f"{dataset_id} - {self.eval_set.capitalize()}"
        ratio_row = [category]
        metric_row = [self.config["model"]]
        prediction_ratio_row = ["prediction_ratio"]

        for key in ordered_keys:
            data_source = fine_grained_results[key]
            ratio = len(data_source) / total_cases
            avg_ndcg = sum(data_source) / len(data_source)
            ratio_row.append(f"{ratio:.2%}")
            metric_row.append(f"{avg_ndcg:.4f}")
            prediction_ratio_row.append(f"{prediction_ratio[key]:.2%}")

        # Write to file
        with open(self.log_file, "a") as f:
            if not has_header:
                f.write("\t".join(header) + "\n")
            f.write("\t".join(ratio_row) + "\n")
            f.write("\t".join(metric_row) + "\n")
            f.write("\t".join(prediction_ratio_row) + "\n")
    
    def _write_inference_results(self, inference_data):
        """Write detailed inference results to CSV file."""
        with open(self.inference_file, 'w', newline='') as f:
            fieldnames = ['sample_id', 'rank_id', 'prediction', 'label', 
                          'ndcg@5', 'ndcg@10', 'recall@5', 'recall@10', 'item_labels']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(inference_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TIGER', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--eval', type=str, default='test', help='Evaluation set, either test or valid')
    parser.add_argument('--log_file', type=str, default=None, help='CSV file path to log results')
    parser.add_argument('--save_inference', action='store_true', help='Save detailed inference results')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    assert args.checkpoint is not None, 'Checkpoint path is required.'
    command_line_configs.update({'epochs': 0})
    command_line_configs.update({'logging': False})

    pipeline = FineGrainedResultPipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_dict=command_line_configs,
        eval_set=args.eval,
        log_file=args.log_file,
        save_inference=args.save_inference,
    )
    pipeline.run()
