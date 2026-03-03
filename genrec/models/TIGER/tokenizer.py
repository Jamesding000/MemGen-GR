import os
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
from genrec.models.TIGER.layers import RQVAEModel
from genrec.utils import list_to_str


class TIGERTokenizer(AbstractTokenizer):
    """
    Tokenizer for the TIGER model.

    An example when "rq_codebook_size == 256, rq_n_codebooks == 3, n_user_tokens == 2000":
        0: padding
        1-256: digit 1
        257-512: digit 2
        513-768: digit 3
        769-1024: digit 4 (used to avoid conflicts)
        1025-3024: user tokens
        3025: eos

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        item2tokens (dict): A dictionary mapping items to their semantic IDs.
        base_user_id (int): The base user ID.
        n_user_tokens (int): The number of user tokens.
        eos_token (int): The end-of-sequence token.
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        super(TIGERTokenizer, self).__init__(config, dataset)

        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        
        self.sem_ids_path = self.config.get('sem_ids_path', os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_{list_to_str(self.codebook_sizes, remove_blank=True)}.sem_ids'
        ))
        
        self.item2tokens = self._init_tokenizer(dataset)
        self.base_user_token = sum(self.codebook_sizes) + 1
        self.n_user_tokens = self.config['n_user_tokens']
        self.eos_token = self.base_user_token + self.n_user_tokens

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """
        Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

        Args:
            dataset (AbstractDataset): The dataset containing the sentences to encode.
            output_path (str): The path to save the encoded sentence embeddings.

        Returns:
            numpy.ndarray: The encoded sentence embeddings.
        """
        assert self.config['metadata'] == 'sentence', \
            'TIGERTokenizer only supports sentence metadata.'

        sent_emb_model = SentenceTransformer(
            self.config['sent_emb_model']
        ).to(self.config['device'])

        meta_sentences = [] # 1-base, meta_sentences[0] -> item_id = 1
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])
        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device']
        )

        sent_embs.tofile(output_path)
        return sent_embs

    def _train_rqvae(self, sent_embs: torch.Tensor, model_path: str) -> RQVAEModel:
        """
        Trains the RQ-VAE model using the given sentence embeddings.

        Args:
            sent_embs (torch.Tensor): Array of sentence embeddings.
            model_path (str): Path to save the trained model.

        Returns:
            rqvae_model: Trained RQ-VAE model.
        """
        device = self.config['device']

        # Initialize RQ-VAE model
        all_hidden_sizes = [sent_embs.shape[1]] + self.config['rqvae_hidden_sizes']
        rqvae_model = RQVAEModel(
            hidden_sizes=all_hidden_sizes,
            n_codebooks=self.config['rq_n_codebooks'],
            codebook_size=self.config['rq_codebook_size'],
            dropout=self.config['rqvae_dropout'],
            low_usage_threshold=self.config['rqvae_low_usage_threshold']
        ).to(device)
        self.log(rqvae_model)
        if os.path.exists(model_path):
            self.log(f"[TOKENIZER] Loading RQ-VAE model from {model_path}...")
            rqvae_model.load_state_dict(torch.load(model_path))
            return rqvae_model

        # Model training
        batch_size = self.config['ravae_batch_size']
        num_epochs = self.config['rqvae_epoch']
        beta = self.config['rqvae_beta']
        verbose = self.config['rqvae_verbose']

        rqvae_model.generate_codebook(sent_embs, device)
        optimizer = torch.optim.Adagrad(rqvae_model.parameters(), lr=self.config['rqvae_lr'])
        train_dataset = TensorDataset(sent_embs)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.log("[TOKENIZER] Training RQ-VAE model...")
        rqvae_model.train()
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0.0
            total_rec_loss = 0.0
            total_quant_loss = 0.0
            total_count = 0
            for batch in dataloader:
                x_batch = batch[0]
                optimizer.zero_grad()
                recon_x, quant_loss, count = rqvae_model(x_batch)
                reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction='mean')
                loss = reconstruction_mse_loss + beta * quant_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().cpu().item()
                total_rec_loss += reconstruction_mse_loss.detach().cpu().item()
                total_quant_loss += quant_loss.detach().cpu().item()
                total_count += count

            if (epoch + 1) % verbose == 0:
                self.log(
                    f"[TOKENIZER] RQ-VAE training\n"
                    f"\tEpoch [{epoch+1}/{num_epochs}]\n"
                    f"\t  Training loss: {total_loss/ len(dataloader)}\n"
                    f"\t  Unused codebook:{total_count/ len(dataloader)}\n"
                    f"\t  Recosntruction loss: {total_rec_loss/ len(dataloader)}\n"
                    f"\t  Quantization loss: {total_quant_loss/ len(dataloader)}\n")
        self.log("[TOKENIZER] RQ-VAE training complete.")

        # Save model
        torch.save(rqvae_model.state_dict(), model_path, pickle_protocol=4)
        return rqvae_model

    def _extend_semantic_ids(self, sem_ids: np.ndarray):
        """
        Extends the semantic IDs from k digits to (k + 1) digits to avoid conflict.

        Args:
            sem_ids (np.ndarray): The input array of semantic IDs.

        Returns:
            dict: A dictionary mapping item IDs to semantic IDs.
        """
        sem_id2item = defaultdict(list)
        item2sem_ids = {}
        max_conflict = 0
        for i in range(sem_ids.shape[0]):
            str_id = ' '.join(map(str, sem_ids[i].tolist()))
            sem_id2item[str_id].append(i + 1)
            item = self.id2item[i + 1]
            item2sem_ids[item] = (*tuple(sem_ids[i].tolist()), len(sem_id2item[str_id]))
            max_conflict = max(max_conflict, len(sem_id2item[str_id]))
        self.log(f'[TOKENIZER] RQ-VAE semantic IDs, maximum conflict: {max_conflict}')
        if max_conflict > self.codebook_sizes[-1]:
            raise ValueError(
                f'[TOKENIZER] RQ-VAE semantic IDs conflict with codebook size: '
                f'{max_conflict} > {self.codebook_sizes[-1]}. Please increase the codebook size.'
            )
        return item2sem_ids

    def _generate_semantic_id(
        self,
        rqvae_model: RQVAEModel,
        sent_embs: torch.Tensor,
        sem_ids_path: str
    ) -> None:
        """
        Generates semantic IDs using the given RQVAE model and saves them to a file.

        Args:
            rqvae_model (RQVAEModel): The RQVAE model used for encoding sentence embeddings.
            sent_embs (torch.Tensor): The sentence embeddings to be encoded.
            sem_ids_path (str): The path to save the generated semantic IDs.

        Returns:
            None
        """
        rqvae_model.eval()
        rqvae_sem_ids = rqvae_model.encode(sent_embs)
        item2sem_ids = self._extend_semantic_ids(rqvae_sem_ids)
        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """
        Get a boolean mask indicating which items are used for training.

        Args:
            dataset (AbstractDataset): The dataset containing the item sequences.

        Returns:
            np.ndarray: A boolean mask indicating which items are used for training.
        """
        items_for_training = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}')
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask

    def _generate_semantic_id_faiss(
        self,
        sent_embs: np.ndarray,
        sem_ids_path: str,
        train_mask: np.ndarray
    ) -> None:
        """
        Generates semantic IDs using the Faiss library and saves them to a file.

        Args:
            sent_embs (np.ndarray): The sentence embeddings.
            sem_ids_path (str): The path to save the semantic IDs.
            train_mask (np.ndarray): A boolean mask indicating which items are used for training.

        Returns:
            None
        """
        n_bits = int(np.log2(self.config['rq_codebook_size']))

        import faiss
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.IndexResidualQuantizer(
            sent_embs.shape[-1],
            self.config['rq_n_codebooks'],
            n_bits,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log(f'[TOKENIZER] Training index...')
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        faiss_sem_ids = []
        uint8_code = index.rq.compute_codes(sent_embs)
        n_bytes = uint8_code.shape[1]
        self.logger.info(f'[TOKENIZER] Generating semantic IDs...')
        for u8_code in uint8_code:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), n_bytes)
            code = []
            for i in range(self.config['rq_n_codebooks']):
                code.append(bs.read(n_bits))
            faiss_sem_ids.append(code)
        faiss_sem_ids = np.array(faiss_sem_ids)
        item2sem_ids = self._extend_semantic_ids(faiss_sem_ids)
        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _generate_atomic_ids(self, dataset: AbstractDataset) -> dict:
        """
        Generates atomic semantic IDs where each item gets its own unique ID.
        Item with item_id=i gets semantic ID [i].

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to their atomic semantic IDs (single digit).
        """
        item2sem_ids = {}
        for item_id in range(1, dataset.n_items):
            item = dataset.id_mapping['id2item'][item_id]
            # Each item gets its own item_id as the semantic ID
            item2sem_ids[item] = (item_id,)
        self.log(f'[TOKENIZER] Generated atomic semantic IDs for {len(item2sem_ids)} items')
        with open(self.sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)
        return item2sem_ids

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """
        Converts semantic IDs to tokens.

        Args:
            item2sem_ids (dict): A dictionary mapping items to their corresponding semantic IDs.

        Returns:
            dict: A dictionary mapping items to their corresponding tokens.
        """
        sem_id_offsets = [0]
        for digit in range(1, self.n_digit):
            sem_id_offsets.append(sem_id_offsets[-1] + self.codebook_sizes[digit - 1])
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                # "+ 1" as 0 is reserved for padding
                tokens[digit] += sem_id_offsets[digit] + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Initialize the tokenizer.

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """
        # Handle atomic IDs case
        if self.use_atomic_ids:
            self.log('[TOKENIZER] Using atomic semantic IDs (naive mode)')
            if not os.path.exists(self.sem_ids_path):
                item2sem_ids = self._generate_atomic_ids(dataset)
            else:
                item2sem_ids = json.load(open(self.sem_ids_path, 'r'))
            item2tokens = self._sem_ids_to_tokens(item2sem_ids)
            return item2tokens

        # Load semantic IDs
        sem_ids_path = self.sem_ids_path

        if not os.path.exists(sem_ids_path):
            # Load or encode sentence embeddings
            sent_emb_path = os.path.join(
                dataset.cache_dir, 'processed',
                f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
            )
            if os.path.exists(sent_emb_path):
                self.log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
                sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings...')
                sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
            # PCA
            if self.config['sent_emb_pca'] > 0:
                self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
            self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

            # Generate semantic IDs
            training_item_mask = self._get_items_for_training(dataset)
            if self.config['rq_faiss']:
                self.log(f'[TOKENIZER] Semantic IDs not found. Training index using Faiss...')
                self._generate_semantic_id_faiss(sent_embs, sem_ids_path, training_item_mask)
            else:
                self.log(f'[TOKENIZER] Semantic IDs not found. Training RQ-VAE model...')
                embs_for_training = torch.FloatTensor(sent_embs[training_item_mask]).to(self.config['device'])
                sent_embs = torch.FloatTensor(sent_embs).to(self.config['device'])
                model_path = os.path.join(
                    dataset.cache_dir, 'processed',
                    f'rqvae_{list_to_str(self.codebook_sizes, remove_blank=True)}.pth'
                )
                rqvae_model = self._train_rqvae(embs_for_training, model_path)
                self._generate_semantic_id(rqvae_model, sent_embs, sem_ids_path)

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        return item2tokens

    @property
    def use_atomic_ids(self):
        """
        Returns True if using atomic IDs (naive semantic IDs where each item gets a unique ID).
        This is enabled when n_codebooks = 1 and codebook_size = -1.
        """
        return (self.config['rq_n_codebooks'] == 1 and 
                self.config['rq_codebook_size'] == -1)

    @property
    def n_digit(self):
        """
        Returns the number of digits for the tokenizer.

        The number of digits is determined by the value of `rq_n_codebooks` in the configuration.
        For atomic IDs, returns 1 (no conflict resolution needed).
        """
        if self.use_atomic_ids:
            return 1
        return self.config['rq_n_codebooks'] + 1

    @property
    def codebook_sizes(self):
        """
        Returns the codebook size for the TIGER tokenizer.

        If `rq_codebook_size` is a list, it returns the list as is.
        If `rq_codebook_size` is -1 (atomic IDs), returns [n_items].
        If `rq_codebook_size` is an integer, it returns a list with `n_digit` elements,
        where each element is equal to `rq_codebook_size`.

        Returns:
            list: The codebook size for the TIGER tokenizer.
        """
        if isinstance(self.config['rq_codebook_size'], list):
            return self.config['rq_codebook_size']
        elif self.config['rq_codebook_size'] == -1:
            # Atomic IDs: codebook size equals number of items
            n_items = len(self.id2item)
            return [n_items]
        else:
            return [self.config['rq_codebook_size']] * self.n_digit

    def _token_single_user(self, user: str) -> int:
        """
        Tokenizes a single user.

        Args:
            user (str): The user to tokenize.

        Returns:
            int: The tokenized user ID.

        """
        user_id = self.user2id[user]
        return self.base_user_token + user_id % self.n_user_tokens

    def _token_single_item(self, item: str) -> int:
        """
        Tokenizes a single item.

        Args:
            item (str): The item to be tokenized.

        Returns:
            list: The tokens corresponding to the item.
        """
        return self.item2tokens[item]

    def _tokenize_once(self, example: dict) -> tuple:
        """
        Tokenizes a single example.

        Args:
            example (dict): A dictionary containing the example data.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, and labels.
        """
        max_item_seq_len = self.config['max_item_seq_len']

        # input_ids
        user_token = self._token_single_user(example['user'])
        input_ids = [user_token]
        for item in example['item_seq'][:-1][-max_item_seq_len:]:
            input_ids.extend(self._token_single_item(item))
        input_ids.append(self.eos_token)
        input_ids.extend([self.padding_token] * (self.max_token_seq_len - len(input_ids)))

        # attention_mask
        item_seq_len = min(len(example['item_seq'][:-1]), max_item_seq_len)
        attention_mask = [1] * (self.n_digit * item_seq_len + 2)
        attention_mask.extend([0] * (self.max_token_seq_len - len(attention_mask)))

        # labels
        labels = list(self._token_single_item(example['item_seq'][-1])) + [self.eos_token]

        return input_ids, attention_mask, labels

    def tokenize_function(self, examples: dict, split: str) -> dict:
        """
        Tokenizes the input example based on the specified split.
        """
        # Initialize lists to store results for the whole batch
        all_input_ids, all_attention_mask, all_labels = [], [], []
        
        # FIX: Iterate by index over the lists inside the dictionary
        batch_size = len(examples['user'])
        
        for i in range(batch_size):
            user = examples['user'][i]
            item_seq = examples['item_seq'][i]

            if split == 'train':
                # Generate multiple training samples from one sequence
                n_return_examples = len(item_seq) - 1
                for j in range(n_return_examples):
                    cur_example = {
                        'user': user,
                        'item_seq': item_seq[:j+2] # Context + Next Target
                    }
                    input_ids, attention_mask, labels = self._tokenize_once(cur_example)
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_labels.append(labels)
            else:
                # Validation/Test: Just one sample per sequence
                cur_example = {
                    'user': user, 
                    'item_seq': item_seq
                }
                input_ids, attention_mask, labels = self._tokenize_once(cur_example)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
               
        return {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'labels': all_labels
        }

    def tokenize(self, datasets: dict) -> dict:
        """
        Tokenizes the given datasets.

        Args:
            datasets (dict): A dictionary of datasets to tokenize.

        Returns:
            dict: A dictionary of tokenized datasets.
        """
        BATCH_SIZE = 128
        tokenized_datasets = {}
        for split in datasets:
            # Add idx only for val/test (1:1 mapping, needed for fine-grained eval)
            if split in ['val', 'test']:
                tokenized_datasets[split] = datasets[split].map(
                    lambda t, idx: {**self.tokenize_function(t, split), 'idx': idx},
                    batched=True,
                    batch_size=BATCH_SIZE,
                    with_indices=True,
                    remove_columns=datasets[split].column_names,
                    num_proc=self.config['num_proc'],
                    desc=f'Tokenizing {split} set: '
                )
            else:
                tokenized_datasets[split] = datasets[split].map(
                    lambda t: self.tokenize_function(t, split),
                    batched=True,
                    batch_size=BATCH_SIZE,
                    remove_columns=datasets[split].column_names,
                    num_proc=self.config['num_proc'],
                    desc=f'Tokenizing {split} set: '
                )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size for the TIGER tokenizer.
        """
        return self.eos_token + 1

    @property
    def max_token_seq_len(self) -> int:
        """
        Returns the maximum token sequence length for the TIGER tokenizer.
        """
        # +2 for user token and eos token
        return self.config['max_item_seq_len'] * self.n_digit + 2
