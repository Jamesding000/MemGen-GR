from logging import getLogger
from typing import Union
import torch
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import get_config, init_seed, init_logger, init_device, \
    get_dataset, get_tokenizer, get_model, get_trainer, log, config_for_log, get_file_name


class Pipeline:
    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        checkpoint_path: str = None,
        tokenizer: AbstractTokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )
        self.checkpoint_path = checkpoint_path

        # Accelerator with wandb
        require_logging = self.config['logging']
        self.accelerator = Accelerator(log_with='wandb' if require_logging else None)
        self.config['accelerator'] = self.accelerator
        self.config['device'] = self.accelerator.device  # use accelerate's device instead of init_device()
        self.config['use_ddp'] = (self.accelerator.num_processes > 1)

        # Seed and Logger
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(f'Device: {self.config["device"]}')
        
        # Initialize wandb tracker
        if require_logging:
            wandb_project = self.config['wandb_project']
            wandb_group = f"{self.config['dataset']}-{self.config['model']}"
            wandb_run_name = self.config['wandb_run_name'] if 'wandb_run_name' in self.config else get_file_name(self.config, suffix='')
            
            self.accelerator.init_trackers(
                project_name=wandb_project,
                config=config_for_log(self.config),
                init_kwargs={
                    "wandb": {
                        "name": wandb_run_name,
                        "group": wandb_group,
                        "tags": [self.config['dataset'], self.config['model']],
                    }
                },
            )

        # Dataset
        self.raw_dataset = get_dataset(dataset_name)(self.config)
        self.log(self.raw_dataset)
        self.split_datasets = self.raw_dataset.split()

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer(self.config, self.raw_dataset)
        else:
            assert isinstance(model_name, str), 'Tokenizer must be provided if model_name is not a string.'
            self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)
        self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)

        # Model
        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.raw_dataset, self.tokenizer)
            if checkpoint_path is not None:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.config['device']))
                self.log(f'Loaded model checkpoint from {checkpoint_path}')
        self.log(self.model)
        self.log(self.model.n_parameters)

        # Trainer
        if trainer is not None:
            self.trainer = trainer
        else:
            self.trainer = get_trainer(model_name)(self.config, self.model, self.tokenizer, self.split_datasets)

    def run(self):
        # DataLoader
        train_dataloader = DataLoader(
            self.tokenized_datasets['train'],
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            collate_fn=self.tokenizer.collate_fn['train']
        )
        val_dataloader = DataLoader(
            self.tokenized_datasets['val'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['val']
        )
        test_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )

        self.trainer.fit(train_dataloader, val_dataloader)

        self.accelerator.wait_for_everyone()
        if self.config['use_ddp']:
            # make sure all processes have the same checkpoint path
            import torch.distributed as dist
            ckpt_path_container = [self.trainer.saved_model_ckpt]
            dist.broadcast_object_list(ckpt_path_container, src=0)
            self.trainer.saved_model_ckpt = ckpt_path_container[0]

        self.model = self.accelerator.unwrap_model(self.model)
        
        if self.config['load_best_ckpt'] and self.checkpoint_path is None:
            self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))
            eval_epoch = self.trainer.best_epoch
            if self.accelerator.is_main_process:
                self.log(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')
        else:
            eval_epoch = self.trainer.last_epoch

        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )

        test_results = self.trainer.evaluate(
            test_dataloader, split='test',
            step=self.trainer.current_step, epoch=eval_epoch)

        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
        self.log(f'Test Results: {test_results}')

        self.trainer.end()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
