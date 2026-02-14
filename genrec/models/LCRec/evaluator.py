import os
import re
from collections import defaultdict, OrderedDict
from logging import getLogger

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from genrec.evaluator import Evaluator
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, PreTrainedTokenizer

from genrec.models.LCRec.dataset import SeqRecDataset
from genrec.models.LCRec.prompts import seqrec_prompts
from genrec.models.LCRec.tokenizer import LCRecTokenizer
from genrec.utils import log

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


class LCRecEvaluator(Evaluator):
    r"""
    Evaluator for LCRec model.

    Args:
        config(dict): Configuration parameters for the model.
        llm_tokenizer(PreTrainedTokenizer): Tokenizer for text.
        item_tokenizer(LCRecTokenizer): Tokenizer for item.
        test_dataset(SeqRecDataset): Test dataset.
    """
    def __init__(self,
             config: dict,
             llm_tokenizer: PreTrainedTokenizer,
             item_tokenizer: LCRecTokenizer,
             test_dataset: SeqRecDataset
        ):
        super(LCRecEvaluator, self).__init__(config, llm_tokenizer)
        self.logger = getLogger()

        self.llm_tokenizer = llm_tokenizer
        self.item_tokenizer = item_tokenizer
        self.test_dataset = test_dataset

        assert self.llm_tokenizer.padding_side == "left"
        self.eos_token = self.llm_tokenizer.eos_token_id

        self.accelerator = self.config['accelerator']
        self.test_epochs = self.config['test_epochs']
        self.test_prompt_ids = self.config['test_prompt_ids']
        if self.test_prompt_ids == "all":
            self.test_prompt_ids = list(range(len(seqrec_prompts)))


        self.model_checkpoint_path = self.config['model_checkpoint_path']
        self.collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            padding="longest",
        )


    def get_all_checkpoint(self, folder):
        content = os.listdir(folder)
        checkpoints = [
            path
            for path in content
            if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
        ]
        checkpoints = sorted(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0]))
        checkpoints = [os.path.join(folder, ckpt) for ckpt in checkpoints]
        return checkpoints


    @torch.no_grad()
    def evaluate(self):

        test_dataloader = DataLoader(self.test_dataset,
                                    batch_size=self.config['eval_batch_size'],
                                    collate_fn=self.collator,
                                    pin_memory=True)

        all_ckpts = self.get_all_checkpoint(self.model_checkpoint_path)
        test_ckpts = [all_ckpts[e-1] for e in self.test_epochs]

        for ckpt in test_ckpts:
            model = self.load_ckpt(ckpt)

            if self.accelerator.is_main_process:
                self.log(f'Loaded model checkpoint from {ckpt}')
                # self.log(model)

            _, mean_results, min_results, max_results = self.evaluate_one_ckpt(model, test_dataloader)

            if self.accelerator.is_main_process:
                for key in mean_results:
                    self.log(f"Mean {key}: {mean_results[key]}")

                for key in min_results:
                    self.log(f"Min {key}: {min_results[key]}")

                for key in max_results:
                    self.log(f"Max {key}: {max_results[key]}")


    def load_ckpt(self, ckpt_path):

        if self.config['use_ddp']:
            device_map = {"": self.accelerator.local_process_index}
            torch.cuda.set_device(self.accelerator.local_process_index)
            # device = torch.device("cuda", self.accelerator.local_process_index)
        else:
            device_map = "auto"
            # device = torch.device("cuda", self.accelerator.local_process_index)

        if self.config['lora']:
            model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name_or_path'],
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
            model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            model = PeftModel.from_pretrained(
                model,
                ckpt_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
                device_map=device_map,
            )

        return model


    def evaluate_one_ckpt(self, model, test_dataloader):

        model, test_dataloader = self.accelerator.prepare(model, test_dataloader)

        model.eval()
        all_prompt_results = []
        for test_prompt_id in self.test_prompt_ids:
            self.log(f"Testing prompt {test_prompt_id}")
            test_dataloader.dataset.set_prompt_id(test_prompt_id)

            test_dataloader_bar = tqdm(
                test_dataloader,
                total=len(test_dataloader),
                desc=f"Eval - Test",
            )

            all_results = defaultdict(list)
            for batch in test_dataloader_bar:
                n_digit = self.item_tokenizer.n_digit
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                # self.log( (batch["input_ids"][0], batch["input_ids"].shape) )
                # self.log( (batch["labels"][0], batch["labels"].shape) )


                if self.config['use_ddp']:  # ddp, gather data from all devices for evaluation

                    outputs = model.module.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens= n_digit + 1,
                        num_beams=self.config['num_beams'],
                        num_return_sequences=self.maxk,
                        output_scores=False,
                        early_stopping=False,
                    )
                    # self.log(outputs[0])
                    B, inputs_len = batch["input_ids"].shape[:2]
                    preds = outputs[:, inputs_len: inputs_len + n_digit].reshape(B, self.maxk, -1)
                    labels = batch["labels"]
                    labels = labels[labels != -100].reshape(B, -1)
                    preds, labels = self.accelerator.gather_for_metrics((preds, labels))
                else:
                    outputs = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=n_digit + 1,
                        num_beams=self.config['num_beams'],
                        num_return_sequences=self.maxk,
                        output_scores=False,
                        early_stopping=False,
                    )
                    B, inputs_len = batch["input_ids"].shape[:2]
                    preds = outputs[:, inputs_len:inputs_len + n_digit].reshape(B, self.maxk, -1)
                    labels = batch["labels"]
                    labels = labels[labels != -100].reshape(B, -1)

                # self.log(preds)
                # self.log(labels)

                results = self.calculate_metrics(preds, labels)
                for key, value in results.items():
                    all_results[key].append(value)

                m_res = OrderedDict()
                for metric in self.config['metrics']:
                    for k in self.config['topk']:
                        key = f"{metric}@{k}"
                        m_res[key] = torch.cat(all_results[key]).mean().item()
                self.log(m_res)

            test_results = OrderedDict()
            for metric in self.config['metrics']:
                for k in self.config['topk']:
                    key = f"{metric}@{k}"
                    test_results[key] = torch.cat(all_results[key]).mean().item()


            if self.accelerator.is_main_process:
                for key in test_results:
                    self.log(f"Prompt_{test_prompt_id} {key}: {test_results[key]}")

            all_prompt_results.append(test_results)

        mean_results = OrderedDict()
        min_results = OrderedDict()
        max_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                all_res = [_[key] for _ in all_prompt_results]
                mean_results[key] = sum(all_res) / len(all_res)
                min_results[key] = min(all_res)
                max_results[key] = max(all_res)

        return all_prompt_results, mean_results, min_results, max_results


    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)



