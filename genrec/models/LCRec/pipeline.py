from logging import getLogger
from typing import Union
import torch
import os

import transformers
from accelerate import Accelerator

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch.distributed as dist
from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.models.LCRec.dataset import get_sft_dataset
from genrec.models.LCRec.evaluator import LCRecEvaluator
from genrec.utils import get_config, init_seed, init_logger, init_device, get_dataset, get_tokenizer, log, get_file_name


class LCRecPipeline:
    r"""
    Pipeline for LCRec model.

    Args:
        model_name(str or AbstractModel): Model name or model class.
        dataset_name(str or AbstractDataset): Dataset name or dataset class.
        tokenizer(AbstractTokenizer): Item Tokenizer.
        config_dict(dict): Configuration parameters for the model.
        config_file(str): Configuration file.
    """
    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        tokenizer: AbstractTokenizer = None,
        config_dict: dict = None,
        config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )

        self.pipeline_stage = self.config['pipeline_stage']

        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device()

        self.config["output_dir"] = os.path.join(
            self.config["ckpt_dir"],
            get_file_name(self.config),
        )

        # Accelerator
        self.project_dir = os.path.join(
            self.config['tensorboard_log_dir'],
            self.config["dataset"],
            self.config["model"]
        )
        self.accelerator = Accelerator(log_with='tensorboard', project_dir=self.project_dir)
        self.config['accelerator'] = self.accelerator


        # Seed and Logger
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(f'Device: {self.config["device"]}')

        # Raw Dataset
        self.raw_dataset = get_dataset(dataset_name)(self.config)
        self.log(self.raw_dataset)
        self.split_datasets = self.raw_dataset.split()


        # Item Tokenizer
        if tokenizer is not None:
            self.item_tokenizer = tokenizer(self.config, self.raw_dataset)
        else:
            assert isinstance(model_name, str), 'Tokenizer must be provided if model_name is not a string.'
            self.item_tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)


        if self.pipeline_stage == "tokenization":
            return
        elif self.pipeline_stage == "training":
            # LLM Tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_name_or_path'],
                use_fast=False,
                padding_side="right",
            )
            if self.llm_tokenizer.pad_token_id is None:
                self.llm_tokenizer.pad_token_id = 0

            added_num = self.llm_tokenizer.add_tokens(self.item_tokenizer.all_tokens)
            self.log(f"Added {added_num} item code tokens")

            if self.accelerator.is_main_process:
                self.llm_tokenizer.save_pretrained(self.config["output_dir"])

            self.llm_collator = DataCollatorForSeq2Seq(
                self.llm_tokenizer,
                pad_to_multiple_of=8,
                padding="longest",
            )

            # SFT Dataset
            self.train_dataset, self.val_dataset, self.test_dataset = get_sft_dataset(self.config, self.raw_dataset, self.item_tokenizer, self.llm_tokenizer)

            if self.config['use_ddp']:
                device_map = {"": self.accelerator.local_process_index}
                torch.cuda.set_device(self.accelerator.local_process_index)
            else:
                device_map = "auto"
            self.config["device_map"] = device_map

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name_or_path'],
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
            self.model.resize_token_embeddings(len(self.llm_tokenizer), mean_resizing=False)
            self.model.config.use_cache = False


            if self.config['lora']:
                lora_config = LoraConfig(
                    r=self.config['lora_r'],
                    lora_alpha=self.config['lora_alpha'],
                    target_modules=self.config['lora_target_modules'],
                    modules_to_save=self.config['lora_modules_to_save'],
                    lora_dropout=self.config['lora_dropout'],
                    bias="none",
                    inference_mode=False,
                    task_type=TaskType.CAUSAL_LM,
                )
                self.model = get_peft_model(self.model, lora_config)


            self.log(self.model)
            self.log(f"Model size: {self.model.num_parameters()}")
            self.log(f"Model trainable parameters: {self.model.num_parameters(only_trainable=True)}")


            # Trainer
            self.trainer = Trainer(
                model= self.model,
                train_dataset= self.train_dataset,
                eval_dataset=self.val_dataset,
                args= TrainingArguments(
                    seed=self.config['rand_seed'],
                    per_device_train_batch_size=self.config['train_batch_size'],
                    per_device_eval_batch_size=self.config['eval_batch_size'],
                    gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
                    warmup_ratio=self.config['warmup_ratio'],
                    num_train_epochs=self.config['epochs'],
                    learning_rate=self.config['lr'],
                    weight_decay=self.config['weight_decay'],
                    lr_scheduler_type="cosine",
                    bf16=True,
                    logging_steps=10,
                    optim="adamw_torch",
                    gradient_checkpointing=True,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    output_dir=self.config["output_dir"],
                    load_best_model_at_end=True,
                    deepspeed=self.config["deepspeed"],
                    report_to=None,
                    save_safetensors=False,
                ),
                tokenizer=tokenizer,
                data_collator=self.llm_collator,
            )
            self.log(self.config['accelerator'].state)

        elif self.pipeline_stage == "evaluation":

            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_checkpoint_path'],
                use_fast=False,
                padding_side="left",
            )

            _, _, self.test_dataset = get_sft_dataset(self.config, self.raw_dataset, self.item_tokenizer, self.llm_tokenizer)
            self.evaluator = LCRecEvaluator(self.config, self.llm_tokenizer, self.item_tokenizer, self.test_dataset)
        else:
            raise NotImplementedError



    def run(self):


        if self.pipeline_stage == "tokenization":
            return
        elif self.pipeline_stage == "training":
            self.trainer.train()
            self.trainer.save_state()
            self.trainer.save_model()
            self.log(f"Final model checkpoint path: {self.config['output_dir']}")
        elif self.pipeline_stage == "evaluation":
            self.evaluator.evaluate()
        else:
            raise NotImplementedError('Invalid pipeline stage')



    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
