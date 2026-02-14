import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import shutil

from genrec.utils import get_pipeline, parse_command_line_args
from torch.utils.data import DataLoader

def run_single_with_config(model_name, dataset_name, hyperparam_config):
    pipeline = get_pipeline(model_name)(
        model_name=model_name,
        dataset_name=dataset_name,
        config_dict=hyperparam_config
    )

    config = pipeline.config
    tokenizer = pipeline.tokenizer

    train_dataloader = DataLoader(
        pipeline.tokenized_datasets['train'],
        batch_size=config['train_batch_size'],
        shuffle=True,
        collate_fn=tokenizer.collate_fn['train']
    )
    val_dataloader = DataLoader(
        pipeline.tokenized_datasets['val'],
        batch_size=config['eval_batch_size'],
        shuffle=False,
        collate_fn=tokenizer.collate_fn['val']
    )
    test_dataloader = DataLoader(
        pipeline.tokenized_datasets['test'],
        batch_size=config['eval_batch_size'],
        shuffle=False,
        collate_fn=tokenizer.collate_fn['test']
    )

    trainer = pipeline.trainer
    trainer.fit(train_dataloader, val_dataloader)

    pipeline.accelerator.wait_for_everyone()
    model = pipeline.accelerator.unwrap_model(pipeline.model)
    if pipeline.checkpoint_path is None:
        model.load_state_dict(torch.load(trainer.saved_model_ckpt))

    model, test_dataloader = pipeline.accelerator.prepare(model, test_dataloader)
    if pipeline.accelerator.is_main_process and pipeline.checkpoint_path is None:
        pipeline.log(f'Loaded best model checkpoint from {trainer.saved_model_ckpt}')

    valid_results = trainer.evaluate(val_dataloader)
    test_results = trainer.evaluate(test_dataloader)

    valid_score = valid_results.get('ndcg@10', 0.0)

    return {
        'best_valid_result': valid_results,
        'best_valid_score': valid_score,
        'test_result': test_results,
        'model_path': trainer.saved_model_ckpt
    }


class HyperParamLoader():
    def __init__(self, arg_range):
        self.arg_range = arg_range
        self.k_list = list(arg_range.keys())
        self.cur_layer = 0
        self.choice = np.zeros((len(self.k_list), 1))
        self.args = []
        self._dfs(0)
        assert len(self.args) == np.prod([len(_) for _ in arg_range.values()])
        self.n_args = len(self.args)
        print('n_args', self.n_args, flush=True)

    def _dfs(self, layer):
        if layer == len(self.k_list):
            ans = {}
            for l, k in enumerate(self.k_list):
                ans[k] = self.arg_range[k][int(self.choice[l])]
            self.args.append(ans)
            return

        k = self.k_list[layer]
        rg = self.arg_range[k]
        for i in range(len(rg)):
            self.choice[layer] = i
            self._dfs(layer + 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SASRec', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--hp', type=str, default='hyper_props/SASRec.json', help='Hyperparameter config file')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    with open(args.hp, 'r') as f:
        arg_range = json.load(f)

    hp_name = os.path.splitext(os.path.basename(args.hp))[0]
    LOG_NAME = os.path.join('hyper_res', f'{args.model}-{args.dataset}-{hp_name}.log')
    os.makedirs('hyper_res', exist_ok=True)
    hlog = open(LOG_NAME, 'a+')
    hlog.write('======= META =======\n')
    hlog.write(str(args) + '\n\n')
    hlog.write(str(arg_range) + '\n\n')
    hlog.flush()

    hp_loader = HyperParamLoader(arg_range)
    best_val_score = best_valid = best_test = best_round = best_params = best_model_path = None

    for j, hyper_params in enumerate(tqdm(hp_loader.args)):
        search_config = command_line_configs.copy()
        search_config.update(hyper_params)

        result = run_single_with_config(args.model, args.dataset, search_config)

        hlog.write(f'======= Round {j} =======\n')
        hlog.write(str(hyper_params) + '\n\n')
        hlog.write('Best Valid Result: ' + str(result['best_valid_result']) + '\n\n')
        hlog.flush()

        if best_val_score is None or result['best_valid_score'] > best_val_score:
            hlog.write(f'\n Best Valid Updated: {best_val_score} -> {result["best_valid_score"]}\n\n')
            hlog.flush()
            best_val_score = result['best_valid_score']
            best_valid = result['best_valid_result']
            best_test = result['test_result']
            best_round = j
            best_params = hyper_params
            best_model_path = result['model_path']

    hlog.write('======= FINAL =======\n')
    hlog.write(f'Best Round: {best_round}\n')
    hlog.write(f'Best Valid Score: {best_val_score}\n')
    hlog.write('Best Params: ' + str(best_params) + '\n')
    hlog.write('Final Valid Result: ' + str(best_valid) + '\n')
    hlog.write('Final Test Result: ' + str(best_test) + '\n')
    hlog.close()

    os.makedirs('saved', exist_ok=True)
    args_str = '-'.join(f'{k}_{v}' for k, v in command_line_configs.items())
    shutil.copy(best_model_path, f'saved/{args.model}-{args.dataset}-{args_str}.pth')