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
from genrec.models.LCRec.layers import RQVAEModel
from genrec.utils import list_to_str


class LCRecTokenizer(AbstractTokenizer):
    """
    Tokenizer for the LCRec model.

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        item2tokens (dict): A dictionary mapping items to their semantic IDs.
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        super(LCRecTokenizer, self).__init__(config, dataset)


        self.id2item = dataset.id_mapping['id2item']
        self.item2tokens = self._init_tokenizer(dataset)

    def _concat_feat(self, item2meta):

        concat_feats = ['description', 'categories', 'features']

        new_item2meta = {}
        for item, meta in item2meta.items():
            description = ''
            for feat in concat_feats:
                description += meta.pop(feat,'') + ' '

            meta['description'] = description.strip()

            new_item2meta[item] = meta


        return new_item2meta

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """
        Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

        Args:
            dataset (AbstractDataset): The dataset containing the sentences to encode.
            output_path (str): The path to save the encoded sentence embeddings.

        Returns:
            numpy.ndarray: The encoded sentence embeddings.
        """
        assert self.config['metadata'] == 'sentence_feature', \
            'LCRecTokenizer only supports sentence_feature metadata.'

        sent_emb_model = SentenceTransformer(
            self.config['sent_emb_model']
        ).to(self.config['device'])


        item2meta = self._concat_feat(dataset.item2meta)
        features_needed = ['title', 'description']

        all_sent_embs = 0
        for feature in features_needed:
            self.log(f"[TOKENIZER] Encoding feature {feature} ...")
            meta_sentences = [] # 1-base, meta_sentences[0] -> item_id = 1
            for i in range(1, dataset.n_items):
                meta_sentences.append(item2meta[dataset.id_mapping['id2item'][i]][feature])
            sent_embs = sent_emb_model.encode(
                meta_sentences,
                convert_to_numpy=True,
                batch_size=self.config['sent_emb_batch_size'],
                show_progress_bar=True,
                device=self.config['device']
            )
            all_sent_embs += sent_embs

        mean_sent_embs = all_sent_embs / len(features_needed)

        mean_sent_embs.tofile(output_path)

        return mean_sent_embs

    @torch.no_grad()
    def _valid_collision(self, rqvae_model, dataloader):

        rqvae_model.eval()
        sem_id_set = set()
        num_sample = 0
        for batch in dataloader:
            x_batch = batch[0]
            num_sample += len(x_batch)
            sem_ids = rqvae_model.encode(x_batch)
            for sem_id in sem_ids:
                sem_id_set.add(str(sem_id.tolist()))

        collision_rate = (num_sample - len(list(sem_id_set))) / num_sample
        rqvae_model.train()
        return collision_rate

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
            low_usage_threshold=self.config['rqvae_low_usage_threshold'],
            sk_epsilon=self.config['rqvae_sk_epsilon'],
            sk_iters=self.config['rqvae_sk_iters']
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
                collision_rate = self._valid_collision(rqvae_model, dataloader)
                self.log(
                    f"[TOKENIZER] RQ-VAE training\n"
                    f"\tEpoch [{epoch+1}/{num_epochs}]\n"
                    f"\t  Training loss: {total_loss/ len(dataloader)}\n"
                    f"\t  Unused codebook:{total_count/ len(dataloader)}\n"
                    f"\t  Recosntruction loss: {total_rec_loss/ len(dataloader)}\n"
                    f"\t  Quantization loss: {total_quant_loss/ len(dataloader)}\n"
                    f"\t  Collision rate: {collision_rate}\n")

        self.log("[TOKENIZER] RQ-VAE training complete.")

        # Save model
        torch.save(rqvae_model.state_dict(), model_path, pickle_protocol=4)
        return rqvae_model

    def _str_sem_ids(self, sem_ids: np.ndarray):

        str_sem_ids = []
        for i in range(sem_ids.shape[0]):
            str_id = str(sem_ids[i].tolist())
            str_sem_ids.append(str_id)

        return np.array(str_sem_ids)

    def _check_collision(self, str_sem_ids):
        tot_item = len(str_sem_ids)
        tot_ids = len(set(str_sem_ids.tolist()))
        self.log(f'[TOKENIZER] Collision rate: {(tot_item - tot_ids) / tot_item}')
        return tot_item == tot_ids

    def _get_collision_items(self, str_sem_ids):
        sem_id2item = defaultdict(list)
        for i, str_sem_id in enumerate(str_sem_ids):
            sem_id2item[str_sem_id].append(i)

        collision_item_groups = []
        for str_sem_id in sem_id2item:
            if len(sem_id2item[str_sem_id]) > 1:
                collision_item_groups.append(sem_id2item[str_sem_id])

        return collision_item_groups

    def _convert_to_dict(self, sem_ids):
        item2sem_ids = {}
        for i, ids in enumerate(sem_ids):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(ids.tolist())

        return item2sem_ids
    def _add_prefix(self, item2sem_ids):
        block = {}
        prefix = [f"<{level}_{block}>" for level in range(1, self.n_digit+1)]
        item2tokens = {}
        for item, sem_ids in item2sem_ids.items():
            ids = [prefix[level].format(int(sem_ids[level])) for level in range(self.n_digit)]
            item2tokens[item] = tuple(ids)

        return item2tokens



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
        str_sem_ids = self._str_sem_ids(rqvae_sem_ids)

        for _ in range(30):
            if self._check_collision(str_sem_ids):
                break
            collision_item_groups = self._get_collision_items(str_sem_ids)
            for collision_items in collision_item_groups:
                d = sent_embs[collision_items].to(self.config['device'])

                new_sem_ids = rqvae_model.encode(d, use_sk=True)
                new_sem_ids = new_sem_ids.reshape(-1, new_sem_ids.shape[-1])
                for item, ids in zip(collision_items, new_sem_ids):

                    rqvae_sem_ids[item] = ids
                    str_sem_ids[item] = str(ids.tolist())

        self._check_collision(str_sem_ids)
        item2sem_ids = self._convert_to_dict(rqvae_sem_ids)


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


    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Initialize the tokenizer.

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """
        # Load semantic IDs
        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_{list_to_str(self.codebook_sizes, remove_blank=True)}.sem_ids'
        )

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

            self.log(f'[TOKENIZER] Semantic IDs not found. Training RQ-VAE model...')
            embs_for_training = torch.FloatTensor(sent_embs[training_item_mask]).to(self.config['device'])
            sent_embs = torch.FloatTensor(sent_embs).to(self.config['device'])
            model_path = os.path.join(dataset.cache_dir, 'processed/rqvae.pth')
            rqvae_model = self._train_rqvae(embs_for_training, model_path)
            self._generate_semantic_id(rqvae_model, sent_embs, sem_ids_path)

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))

        item2tokens = self._add_prefix(item2sem_ids)

        return item2tokens

    @property
    def n_digit(self):
        """
        Returns the number of digits for the tokenizer.

        The number of digits is determined by the value of `rq_n_codebooks` in the configuration.
        """
        return self.config['rq_n_codebooks']

    @property
    def codebook_sizes(self):
        """
        Returns the codebook size for the LCRec tokenizer.

        If `rq_codebook_size` is a list, it returns the list as is.
        If `rq_codebook_size` is an integer, it returns a list with `n_digit` elements,
        where each element is equal to `rq_codebook_size`.

        Returns:
            list: The codebook size for the LCRec tokenizer.
        """
        if isinstance(self.config['rq_codebook_size'], list):
            return self.config['rq_codebook_size']
        else:
            return [self.config['rq_codebook_size']] * self.n_digit

    @property
    def all_tokens(self):
        """
        Returns all tokens in the tokenizer.
		"""
        all_tokens = set()
        for item in self.item2tokens:
            all_tokens.update(self.item2tokens[item])

        all_tokens = sorted(list(all_tokens))
        return all_tokens
    def token_single_item(self, item: str):
        """
        Tokenizes a single item.

        Args:
            item (str): The item to be tokenized.

        Returns:
            list: The tokens corresponding to the item.
        """
        return "".join(self.item2tokens[item])

    def __call__(self, item: str):

        return  self.token_single_item(item)








