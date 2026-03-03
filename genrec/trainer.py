import os
import csv
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict
from logging import getLogger
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_scheduler
from accelerate import Accelerator

from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.evaluator import Evaluator
from mem_gen_categorizer import FineGrainedEvaluator
from genrec.utils import get_file_name, get_total_steps, config_for_log, log


class Trainer:
    """
    A class that handles the training process for a model.

    Args:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the data.

    Attributes:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        evaluator (Evaluator): The evaluator used for evaluating the model.
        logger (Logger): The logger used for logging training progress.
        accelerator (Accelerator): The accelerator used for distributed training.
        saved_model_ckpt (str): The file path for saving the trained model checkpoint.

    Methods:
        fit(train_dataloader, val_dataloader): Trains the model using the provided training and validation dataloaders.
        evaluate(dataloader, split='test'): Evaluate the model on the given dataloader.
        end(): Ends the training process and releases any used resources.
    """

    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer, split_datasets: dict = None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            get_file_name(self.config, suffix='.pth')
        )
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)
        
        # Initialize fine-grained evaluator if enabled
        self.do_fine_grained_eval = self.config.get('eval_fine_grained', False) and split_datasets is not None
        
        if self.do_fine_grained_eval:
            self.fine_grained_metric = self.config.get('fine_grained_metric', 'ndcg@10')
            self.max_hop = self.config.get('max_hop', 4)
            self.fine_grained_evaluator = FineGrainedEvaluator(
                train_item_seqs=split_datasets['train']['item_seq'],
                max_hop=self.max_hop
            )
            self.split_datasets = split_datasets
            
            # Build reverse mapping for (Tokens -> ItemID)
            assert hasattr(self.tokenizer, 'item2tokens'), "Tokenizer must have item2tokens attribute"
            self.tokens2item = {v if isinstance(v, int) else tuple(v): k for k, v in self.tokenizer.item2tokens.items()}
            
            # Pre-compute labels and ratios for all splits
            self.case_labels = {}
            self.fine_grained_ratios = {}
            for split in ['val', 'test']:
                self.case_labels[split] = {}
                for idx, item_seq in enumerate(self.split_datasets[split]['item_seq']):
                    self.case_labels[split][idx] = self.fine_grained_evaluator.get_case_labels(item_seq)
                
                # Compute static ratios using evaluator
                self.fine_grained_ratios[split] = self.fine_grained_evaluator.compute_pattern_statistics(
                    self.split_datasets[split]['item_seq']
                )
            
        # Set up CSV file for logging evaluation results
        log_dir = os.path.join(self.config['log_dir'], 'fine_grained_results')
        os.makedirs(log_dir, exist_ok=True)
        self.eval_results_file = self.config['eval_results_file'] or os.path.join(
            log_dir,
            get_file_name(self.config, suffix='_eval_results.csv')
        )

    def _log_fine_grained_ratio_table(self):
        """Print the static fine-grained pattern ratios as a pandas DataFrame."""        
        import pandas as pd
        
        # Create DataFrame with ratios
        patterns = list(self.fine_grained_evaluator.logic2judger.keys()) + ['uncategorized']
        data = {}
        
        for split in ['val', 'test']:
            split_name = "Valid" if split == "val" else "Test"
            data[split_name] = {}
            
            for hop in range(1, self.max_hop + 1):
                for pattern in patterns:
                    label = f"{pattern}_{hop}"
                    col_name = f"Hop{hop}_{pattern}"
                    ratio = self.fine_grained_ratios[split].get(label, 0.0)
                    data[split_name][col_name] = f"{ratio * 100:.2f}%"
        
        # Create and print DataFrame
        df = pd.DataFrame(data).T
        self.log("\n" + "="*80)
        self.log("Fine-Grained Pattern Ratios")
        self.log("="*80)
        self.log("\n" + df.to_string())
        self.log("="*80 + "\n")
    
    def fit(self, train_dataloader, val_dataloader):
        """
        Trains the model using the provided training and validation dataloaders.

        Args:
            train_dataloader: The dataloader for training data.
            val_dataloader: The dataloader for validation data.
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'] * (self.accelerator.num_processes ** 0.5),  # square root scaling for DDP
            weight_decay=self.config['weight_decay']
        )

        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'] // self.accelerator.num_processes,  # Adjusted for DDP
            num_training_steps=total_n_steps // self.accelerator.num_processes,  # Adjusted for DDP
        )

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        
        # Log static fine-grained ratios at the start of training
        if self.do_fine_grained_eval and self.accelerator.is_main_process:
            self._log_fine_grained_ratio_table()
            # self._initialize_eval_results_csv()

        n_epochs = np.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)).astype(int)
        self.best_epoch = 0
        best_val_score = -1
        self.current_step = 0
        
        # If budget_epochs is specified, use it instead of n_epochs
        budget_epochs = self.config.get('budget_epochs', None)
        n_epochs = budget_epochs or n_epochs
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
            )
            for batch in train_progress_bar:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.item()
                self.current_step += 1

            self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)
            self.log(f'[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_dataloader)}')

            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:
                all_results = self.evaluate(val_dataloader, split='val', step=self.current_step, epoch=epoch + 1)
                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log({f"Val_Metric/{key}": all_results[key]}, step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')
                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    self.best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        if self.config['use_ddp']: # unwrap model for saving
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)
                        self.log(f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}')

                if self.config['patience'] is not None and epoch + 1 - self.best_epoch >= self.config['patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')
                    break
        self.last_epoch = epoch + 1
        self.log(f'Best epoch: {self.best_epoch}, Best val score: {best_val_score}')
        
        # Log evaluation results as wandb artifact
        if self.do_fine_grained_eval and self.accelerator.is_main_process:
            self._log_eval_results_artifact()

    # def _initialize_eval_results_csv(self):
    #     """Initialize the CSV file for logging evaluation results."""
    #     with open(self.eval_results_file, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         # Header: step, epoch, split, then all metrics
    #         header = ['step', 'epoch', 'split']
    #         # Add standard metrics
    #         for metric in self.config['metrics']:
    #             for k in self.config['topk']:
    #                 header.append(f"{metric}@{k}")
    #         # Add fine-grained metrics
    #         for hop in range(1, self.max_hop + 1):
    #             for logic in list(self.fine_grained_evaluator.logic2judger.keys()) + ['uncategorized']:
    #                 header.append(f"FG/{logic}_{hop}")
    #         writer.writerow(header)
    
    def _append_eval_results_to_csv(self, results, step, epoch, split):
        """Append evaluation results to the CSV file."""
        
        file_exists = os.path.exists(self.eval_results_file)
        with open(self.eval_results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['step', 'epoch', 'split'] + list(results.keys()))
            
            row = [step, epoch, split] + list(results.values())
            writer.writerow(row)
        
    def _log_eval_results_artifact(self):
        """Log the evaluation results CSV as a wandb artifact."""
        import wandb
        artifact = wandb.Artifact(
            name="eval_results",
            type="evaluation_results",
            description="Fine-grained evaluation results for all evaluation steps"
        )
        artifact.add_file(self.eval_results_file)
        wandb.run.log_artifact(artifact)
        self.log(f"Logged evaluation results artifact: {self.eval_results_file}")
    
    def evaluate(self, dataloader, split='test', step=None, epoch=None):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        if self.do_fine_grained_eval:
            test_cases = self.split_datasets[split]['item_seq']
            case_labels = self.case_labels[split]
                
            fine_grained_results = {label: [] for label in self.fine_grained_evaluator.ordered_keys}
            prediction_category_counts = {label: 0 for label in self.fine_grained_evaluator.ordered_keys}
            total_predictions_count = 0
            
        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
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

                batch_results = self.evaluator.calculate_metrics(all_preds, all_labels)
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
                            pred_item_categories = {'uncategorized'}
                        else:
                            history_seq = test_cases[case_idx][:-1]
                            pred_item_categories = self.fine_grained_evaluator.get_case_labels(history_seq + [pred_item_id])
                        
                        for cat in pred_item_categories:
                            prediction_category_counts[cat] += 1
                            
                        total_predictions_count += 1
        
        # print('all_results', all_results)
        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()
        
        if self.do_fine_grained_eval:
            prediction_ratio = {}
            total_cases = len(test_cases)
            for label_key in self.fine_grained_evaluator.ordered_keys:
                if fine_grained_results[label_key]:
                    avg_ndcg = sum(fine_grained_results[label_key]) / len(fine_grained_results[label_key])
                    ratio = len(fine_grained_results[label_key]) / total_cases
                    fine_grained_results[label_key] = avg_ndcg
                else:
                    fine_grained_results[label_key] = np.nan

                count = prediction_category_counts[label_key]
                prediction_ratio[label_key] = count / total_predictions_count
            
        # if self.do_fine_grained_eval:
            # batch_indices = batch_indices.cpu().tolist()
            # scores = results[self.fine_grained_metric]
            # scores = scores.cpu().tolist()
            # preds = preds.cpu().tolist()
            
            # fine_grained_results = self.fine_grained_evaluator.compute_fine_grained_metrics(
            #     batch_indices, 
            #     test_cases, 
            #     scores,
            #     case_labels=self.case_labels[split]
            # )
            # print(fine_grained_results)
            # prediction_ratio = self.fine_grained_evaluator.compute_prediction_distribution(
            #     batch_indices, 
            #     test_cases, 
            #     preds, 
            #     self.tokens2item
            # )
            # print(prediction_ratio)

            for key in fine_grained_results.keys():
                output_results[f"FG/{key}"] = fine_grained_results[key]
            
            for key in prediction_ratio.keys():
                output_results[f"prediction_ratio/{key}"] = prediction_ratio[key]
            
        # Save to CSV if step/epoch provided (during training)
        if step is not None and epoch is not None and self.accelerator.is_main_process:
            self._append_eval_results_to_csv(output_results, step, epoch, split)
    
        return output_results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
    