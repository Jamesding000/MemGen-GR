"""
SASRec Confidence Extraction Script

Extracts confidence scores (MSP, Negative Entropy) from SASRec model predictions.
"""
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import sys
import os
import numpy as np
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genrec.utils import parse_command_line_args
from genrec.pipeline import Pipeline
from torch.utils.data import DataLoader


def extract_category_from_checkpoint(checkpoint_path):
    """
    Extract category/version information from checkpoint filename.
    
    Supports formats:
    - SASRec-AmazonReviews2014-category_Sports_and_Outdoors.pth -> Sports_and_Outdoors
    - genrec_default---model=SASRec_--category=Industrial_and_Scientific-Jan-12-2026.pth -> Industrial_and_Scientific
    - SASRec-Yelp-version_Yelp_2020.pth -> Yelp_2020
    - SASRec-Steam.pth -> None
    """
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


class SASRecConfidencePipeline(Pipeline):
    """
    Extract confidence scores from SASRec model
    """
    def __init__(self, eval_set='test', category=None, version=None, n_predictions=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_set = eval_set
        self.category = category
        self.version = version
        self.n_predictions = n_predictions
    
    def _calculate_ndcg(self, target, predictions, k=10):
        """Calculate NDCG@K for a single sample"""
        if target in predictions[:k]:
            rank = predictions.index(target)
            return 1.0 / np.log2(rank + 2)
        return 0.0

    def generate_with_confidence(self, batch, n_return_sequences=10):
        """
        Generate predictions with confidence scores.
        Key improvements: Top-K normalized softmax and negative entropy for confidence.
        """
        outputs = self.model.gpt2(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        logits = self.model.gather_index(outputs.logits, batch['seq_lens'] - 1)
        probs = torch.softmax(logits, dim=-1)
        
        max_probs, _ = torch.max(probs, dim=-1)
        neg_entropy = torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        
        k_for_norm = 10
        topk_logits, _ = logits.topk(k_for_norm, dim=-1)
        topk_probs_norm = torch.softmax(topk_logits, dim=-1)
        max_probs_topk_norm, _ = torch.max(topk_probs_norm, dim=-1)
        
        topk_preds = logits.topk(n_return_sequences, dim=-1)

        return {
            'predictions': topk_preds.indices,
            'scores': topk_preds.values,
            'max_probability': max_probs,
            'max_probability_norm': max_probs_topk_norm,
            'neg_entropy': neg_entropy,
        }

    def run(self):
        """Run inference on test set and save confidence information."""
        self.log("SASRec Confidence Extraction")
        self.log(f"Dataset: {self.config['dataset']}")
        if self.category:
            self.log(f"Category: {self.category}")
        if self.version:
            self.log(f"Version: {self.version}")

        eval_dataloader = DataLoader(
            self.tokenized_datasets[self.eval_set],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn[self.eval_set]
        )
        self.model, eval_dataloader = self.accelerator.prepare(self.model, eval_dataloader)

        results = []
        self.model.eval()
        n = self.n_predictions

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Extracting confidence")):
                outputs = self.generate_with_confidence(batch, n_return_sequences=n)

                batch_size = len(batch['labels'])
                for i in range(batch_size):
                    pred_item = outputs['predictions'][i, 0].item()
                    target_item = batch['labels'][i].item()
                    is_correct = (pred_item == target_item)

                    top_preds = outputs['predictions'][i, :n].cpu().tolist()
                    top_scores = outputs['scores'][i, :n].cpu().tolist()
                    ndcg_5 = self._calculate_ndcg(target_item, top_preds, k=5)
                    ndcg_10 = self._calculate_ndcg(target_item, top_preds, k=10)
                    recall_5 = 1.0 if target_item in top_preds[:5] else 0.0
                    recall_10 = 1.0 if target_item in top_preds else 0.0

                    results.append({
                        'sample_idx': batch_idx * self.config['eval_batch_size'] + i,
                        'user_id': batch_idx * self.config['eval_batch_size'] + i,
                        'target_item': target_item,
                        'pred_item': pred_item,
                        'top_items': top_preds,
                        'top_scores': top_scores,
                        'confidence_msp': outputs['max_probability'][i].item(),
                        'confidence_msp_norm': outputs['max_probability_norm'][i].item(),
                        'confidence_neg_entropy': outputs['neg_entropy'][i].item(),
                        'is_correct': int(is_correct),
                        'ndcg@5': ndcg_5,
                        'ndcg@10': ndcg_10,
                        'recall@5': recall_5,
                        'recall@10': recall_10,
                    })

        df = pd.DataFrame(results)
        dataset_identifier = get_dataset_identifier(
            self.config['dataset'],
            category=self.category,
            version=self.version
        )
        output_path = f'Confidence/SASRec/results/sasrec_predictions_with_scores_{self.eval_set}_{dataset_identifier}.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        self.log(f"Results saved to: {output_path}")
        self.log(f"Total samples: {len(df)}, NDCG@10: {df['ndcg@10'].mean():.4f}, Recall@10: {df['recall@10'].mean():.4f}")

        self.trainer.end()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract confidence from SASRec")
    parser.add_argument('--model', type=str, default='SASRec', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--eval', type=str, default='test', help='Evaluation set')

    parser.add_argument('--category', type=str, default=None,
                        help='Dataset category. If not specified, will extract from checkpoint filename.')
    parser.add_argument('--version', type=str, default=None,
                        help='Dataset version (for Yelp dataset). If not specified, will extract from checkpoint filename.')
    parser.add_argument('--n_predictions', type=int, default=10,
                        help='Number of top predictions to save per sample (default: 10)')

    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    extracted_info = extract_category_from_checkpoint(args.checkpoint)
    if 'Yelp' in args.dataset:
        if args.version is None and extracted_info:
            args.version = extracted_info
    else:
        if args.category is None and extracted_info:
            args.category = extracted_info
    
    if args.category is not None:
        command_line_configs['category'] = args.category
    if args.version is not None:
        command_line_configs['version'] = args.version
    command_line_configs.update({'epochs': 0, 'logging': False})

    pipeline = SASRecConfidencePipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_dict=command_line_configs,
        eval_set=args.eval,
        category=args.category,
        version=args.version,
        n_predictions=args.n_predictions
    )
    pipeline.run()
