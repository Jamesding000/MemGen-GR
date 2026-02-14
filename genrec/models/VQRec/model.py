import torch
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class VQRec(AbstractModel):
    """
    VQ-Rec model from Hou et al. "Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders." WWW 2023.

    Args:
        config (dict): The configuration of the model.
        dataset (AbstractDataset): The dataset.
        tokenizer (AbstractTokenizer): The tokenizer.

    Attributes:
        item_id2tokens (torch.Tensor): The item ID to tokens mapping.
        gpt2 (GPT2Model): The GPT-2 backbone model.
        loss_fct (torch.nn.CrossEntropyLoss): The loss function.
    """
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(VQRec, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

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

        self.gpt2 = GPT2Model(gpt2config)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

    def _map_item_tokens(self) -> torch.Tensor:
        """
        Maps item tokens to their corresponding item IDs.

        Returns:
            item_id2tokens (torch.Tensor): A tensor of shape (n_items, n_digit) where each row represents the semantic IDs of an item.
        """
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

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

    def get_item_embeddings(self) -> torch.Tensor:
        """
        Returns the embeddings of the items.

        Returns:
            torch.Tensor: The embeddings of the items.
        """
        return self.gpt2.wte.weight[self.item_id2tokens].mean(dim=-2)

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        """
        Forward pass of the model. Returns the logits and the loss.

        Args:
            batch (dict): The input batch.
            return_loss (bool): Whether to return the loss.

        Returns:
            outputs (ModelOutput): 
                The output of the model, which includes:
                - loss (torch.Tensor)
                - logits (torch.Tensor)
        """
        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask']
        )
        item_embs = self.get_item_embeddings()
        outputs.logits = (outputs.last_hidden_state @ item_embs.T)
        if return_loss:
            assert 'labels' in batch, 'The batch must contain the labels.'
            labels = batch['labels'].view(-1)
            outputs.loss = self.loss_fct(outputs.logits.view(-1, self.dataset.n_items), labels)
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
        outputs = self.forward(batch, return_loss=False)
        logits = self.gather_index(outputs.logits, batch['seq_lens'] - 1)
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return preds.unsqueeze(-1)
