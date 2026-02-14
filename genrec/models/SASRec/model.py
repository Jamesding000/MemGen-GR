import torch
from transformers import GPT2Config, GPT2LMHeadModel

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class SASRec(AbstractModel):
    """
    SASRec model from Wang and McAuley, "Self-Attentive Sequential Recommendation." ICDM 2018.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (AbstractDataset): The dataset object.
        tokenizer (AbstractTokenizer): The tokenizer object.

    Attributes:
        gpt2 (GPT2LMHeadModel): The GPT-2 model used for the SASRec model.
    """
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(SASRec, self).__init__(config, dataset, tokenizer)

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )

        self.gpt2 = GPT2LMHeadModel(gpt2config)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

    @property
    def n_parameters(self) -> str:
        """
        Get the number of parameters in the model.

        Returns:
            str: A string representation of the number of parameters in the model.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass of the model. Returns the logits and the loss.

        Args:
            batch (dict): The input batch.

        Returns:
            outputs (ModelOutput): 
                The output of the model, which includes:
                - loss (torch.Tensor)
                - logits (torch.Tensor)
        """
        outputs = self.gpt2(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        logits = outputs.logits.view(-1, outputs.logits.shape[-1])
        labels = batch['labels'].view(-1)
        outputs.loss = self.loss_fct(logits, labels)
        return outputs

    def gather_index(self, output, index):
        """
        Gather the output at a specific index.

        Args:
            output: The output tensor.
            index: The index tensor.

        Returns:
            torch.Tensor: The gathered output.
        """
        index = index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=index).squeeze(1)

    def generate(self, batch, n_return_sequences=1):
        """
        Generate sequences based on the input batch.

        Args:
            batch: The input batch.
            n_return_sequences (int): The number of sequences to generate.

        Returns:
            torch.Tensor: The generated sequences.
        """
        outputs = self.gpt2(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = self.gather_index(outputs.logits, batch['seq_lens'] - 1)
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return preds.unsqueeze(-1)
