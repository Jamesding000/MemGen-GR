import random

from torch.utils.data import Dataset, ConcatDataset
from transformers import PreTrainedTokenizer

from genrec.dataset import AbstractDataset
from genrec.models.LCRec.prompts import seqrec_prompts, item2index_prompts, index2item_prompts
from genrec.models.LCRec.tokenizer import LCRecTokenizer


def get_sft_dataset(config: dict, raw_dataset: AbstractDataset, item_tokenizer: LCRecTokenizer, llm_tokenizer: PreTrainedTokenizer):

    train_seqrec_dataset = SeqRecDataset(config, raw_dataset, item_tokenizer, llm_tokenizer, seqrec_prompts, split='train')
    item2index_dataset = ItemFeatDataset(config, raw_dataset, item_tokenizer, llm_tokenizer, item2index_prompts)
    index2item_dataset = ItemFeatDataset(config, raw_dataset, item_tokenizer, llm_tokenizer, index2item_prompts)

    train_dataset = ConcatDataset([train_seqrec_dataset, item2index_dataset, index2item_dataset])
    val_dataset = SeqRecDataset(config, raw_dataset, item_tokenizer, llm_tokenizer, seqrec_prompts, split='val')
    test_dataset = SeqRecDataset(config, raw_dataset, item_tokenizer, llm_tokenizer, seqrec_prompts, split='test')

    return train_dataset, val_dataset, test_dataset


class SFTDataset(Dataset):
    r"""
    Base dataset class for supervised fine-tuning.

    Args:
        config(dict): Configuration parameters for the model.
        raw_dataset(AbstractDataset): Raw dataset.
        item_tokenizer(LCRecTokenizer): Tokenizer for item.
        llm_tokenizer(PreTrainedTokenizer): Tokenizer for text.
    """
    def __init__(self,
                 config: dict,
                 raw_dataset: AbstractDataset,
                 item_tokenizer: LCRecTokenizer,
                 llm_tokenizer: PreTrainedTokenizer,
                 prompts
        ):
        super(SFTDataset, self).__init__()

        self.config = config
        self.raw_dataset = raw_dataset
        if self.raw_dataset.split_data is None:
            self.raw_dataset.split()
        self.item_tokenizer = item_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.prompts = prompts
        self.prompt_id = None

        self.data = self._get_sft_data()
        
    def _get_sft_data(self):
        
        raise NotImplementedError('Method _get_sft_data() must be implemented in subclass.')

    def set_prompt_id(self, prompt_id):
        self.prompt_id = prompt_id
        
    def __len__(self):
        return len(self.data)

    def _get_inputs_data(self, example, prompt):

        instruction = prompt["instruction"].format(**example)
        target = prompt["target"].format(**example)
        
        a_ids = self.llm_tokenizer.encode(text=instruction, add_special_tokens=True, truncation=True,
                                      max_length=self.config["max_source_length"])
        b_ids = self.llm_tokenizer.encode(text=target, add_special_tokens=False, truncation=True,
                                      max_length=self.config["max_target_length"])

        # Fix the possible bug of leading space token when using LlaMA 2
        if len(b_ids) > 0 and b_ids[0] == self.llm_tokenizer.convert_tokens_to_ids(" "):
            b_ids = b_ids[1:]

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.llm_tokenizer.eos_token_id]
        labels = [-100] * context_length + b_ids + [self.llm_tokenizer.eos_token_id]

        return input_ids, labels


    def __getitem__(self, index):

        example = self.data[index]

        if self.prompt_id is not None:
            prompt = self.prompts[self.prompt_id]
        else:
            prompt_id = random.randint(0, len(self.prompts) - 1)
            prompt = self.prompts[prompt_id]

        input_ids, labels = self._get_inputs_data(example, prompt)

        return dict(input_ids=input_ids, labels=labels)


class SeqRecDataset(SFTDataset):
    r"""
    Dataset class for sequential recommendation.

    Args:
        config(dict): Configuration parameters for the model.
        raw_dataset(AbstractDataset): Raw dataset.
        item_tokenizer(LCRecTokenizer): Tokenizer for item.
        llm_tokenizer(PreTrainedTokenizer): Tokenizer for text.
        split(str): Split of the dataset. Default: 'train'.
    """
    def __init__(self,
                 config: dict,
                 raw_dataset: AbstractDataset,
                 item_tokenizer: LCRecTokenizer,
                 llm_tokenizer: PreTrainedTokenizer,
                 prompts,
                 split='train'
    ):
        self.split = split
        if self.split == "valid":
            self.split = "val"
        super(SeqRecDataset, self).__init__(config, raw_dataset, item_tokenizer, llm_tokenizer, prompts)

    def _get_sft_data(self):
        if self.split == 'train':
            seqrec_data = self._process_train_data()
        else:
            seqrec_data = self._process_val_test_data()

        return seqrec_data

    def _process_train_data(self):

        item_seqs = self.raw_dataset.split_data['train']['item_seq']
        seqrec_data = []
        for item_seq in item_seqs:
            tokenized_item_seq = [self.item_tokenizer(item) for item in item_seq]
            for i in range(1, len(tokenized_item_seq)):
                example = dict()
                example["item"] = tokenized_item_seq[i]
                history = tokenized_item_seq[:i]
                history = history[-self.config["max_item_seq_len"]:]
                example["inters"] = self.config["his_sep"].join(history)
                seqrec_data.append(example)

        return seqrec_data

    def _process_val_test_data(self):

        item_seqs = self.raw_dataset.split_data[self.split]['item_seq']
        seqrec_data = []
        for item_seq in item_seqs:
            tokenized_item_seq = [self.item_tokenizer(item) for item in item_seq]
            example = dict()
            example["item"] = tokenized_item_seq[-1]
            history = tokenized_item_seq[:-1]
            history = history[-self.config["max_item_seq_len"]:]
            example["inters"] = self.config["his_sep"].join(history)
            seqrec_data.append(example)

        return seqrec_data

    def _prepare_generation_inputs(self, inputs):
        input_ids, labels = inputs["input_ids"], inputs["labels"]

        new_input_ids = []
        new_labels = []
        for token, label in zip(input_ids, labels):
            if label == -100:
                new_input_ids.append(token)
            else:
                new_labels.append(label)


        return dict(input_ids=new_input_ids, labels=new_labels)

    def __getitem__(self, index):

        inputs = super().__getitem__(index)

        if self.split == "test":
            inputs = self._prepare_generation_inputs(inputs)

        return inputs





class ItemFeatDataset(SFTDataset):
    r"""
    Dataset class for explicit language-index alignment.
    """
    def _get_sft_data(self):
        items_for_training = set()
        for item_seq in self.raw_dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
                
        item_meta = []
        for item in items_for_training:
            meta = self.raw_dataset.item2meta[item]
            meta['item'] = self.item_tokenizer(item)
            item_meta.append(meta)
            
        return item_meta
    
    