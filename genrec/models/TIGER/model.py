import torch
from transformers import T5Config, T5ForConditionalGeneration

from genrec.model import AbstractModel
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class TIGER(AbstractModel):
    """
    TIGER model from Rajput et al. "Recommender Systems with Generative Retrieval." NeurIPS 2023.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (AbstractDataset): The dataset object.
        tokenizer (AbstractTokenizer): The tokenizer object.

    Attributes:
        t5 (T5ForConditionalGeneration): The T5 model for conditional generation.
    """
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(TIGER, self).__init__(config, dataset, tokenizer)

        t5config = T5Config(
            num_layers=config['num_layers'], 
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            activation_function=config['activation_function'],
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.padding_token,
            eos_token_id=tokenizer.eos_token,
            decoder_start_token_id=0,
            feed_forward_proj=config['feed_forward_proj'],
            n_positions=tokenizer.max_token_seq_len,
        )

        self.t5 = T5ForConditionalGeneration(config=t5config)

    @property
    def n_parameters(self) -> str:
        """
        Calculates the number of trainable parameters in the model.

        Returns:
            str: A string containing the number of embedding parameters, non-embedding parameters, and total trainable parameters.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.t5.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass of the model. Returns the output logits and the loss value.

        Args:
            batch (dict): A dictionary containing the input data for the model.

        Returns:
            outputs (ModelOutput): 
                The output of the model, which includes:
                - loss (torch.Tensor)
                - logits (torch.Tensor)
        """
        outputs = self.t5(**batch)
        return outputs

    def generate(self, batch: dict, n_return_sequences: int = 1, return_scores: bool = False):
        """
        Generates sequences using beam search algorithm.

        Args:
            batch (dict): A dictionary containing input_ids and attention_mask.
            n_return_sequences (int): The number of sequences to generate.
            return_scores (bool): Whether to return the beam scores (log probs).

        Returns:
            torch.Tensor or dict: The generated sequences, or a dict if return_scores=True.
        """
        n_digit = self.tokenizer.n_digit
        num_beams = self.config['num_beams']
        batch_size = batch['input_ids'].shape[0]

        with torch.no_grad():
            outputs = self.t5.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=n_digit + 1,
                num_beams=num_beams,
                num_return_sequences=n_return_sequences,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )

        # 1. Process Sequences
        sequences = outputs.sequences
        seq_len = sequences.shape[1]
        
        # Reshape to (batch_size, n_return_sequences, seq_len)
        sequences = sequences.reshape(batch_size, n_return_sequences, seq_len)
        
        # Extract semantic tokens (skip decoder_start_token)
        pred_tokens = sequences[:, :, 1:1+n_digit]

        # 2. Return Logic (Minimal Change)
        if return_scores:
            # sequences_scores contains the sum of log probs for the generated sequence
            # Shape: (batch_size * n_return_sequences,) -> Reshape to (batch_size, n_return_sequences)
            scores = outputs.sequences_scores.reshape(batch_size, n_return_sequences)
            return {
                'preds': pred_tokens, 
                'scores': scores
            }
        
        return pred_tokens