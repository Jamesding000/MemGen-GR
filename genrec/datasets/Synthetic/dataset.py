import os
import gzip
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from typing import Optional

from genrec.dataset import AbstractDataset
from genrec.utils import download_file, clean_text


class Synthetic(AbstractDataset):
    """
    A class representing the synthetic dataset.

    Args:
        config (dict): A dictionary containing the configuration parameters for the dataset.

    Attributes:
        config (dict): A dictionary containing the configuration parameters for the dataset.
        logger (Logger): An instance of the logger for logging information.
        category (str): The category of the dataset.
        cache_dir (str): The directory path for caching the dataset.
        all_item_seqs (dict): A dictionary containing all the user-item sequences.
        id_mapping (dict): A dictionary containing data maps.
        item2meta (dict): A dictionary containing the item metadata.
    """

    def __init__(self, config: dict):
        super(Synthetic, self).__init__(config)

        self.mode = config['sync_mode']
        self.log(
            f'[DATASET] Synthetic data in mode: {self.mode}'
        )

        self.cache_dir = os.path.join(
            config['cache_dir'], 'Synthetic', self.mode
        )
        self._download_and_process_raw()

    def _load_reviews(self, path: str) -> list:
        """
        Load reviews from a given path.

        Args:
            path (str): The path to the file containing the reviews.

        Returns:
            list: A list of tuples representing the reviews. Each tuple contains the user ID, item ID, and the interaction timestamp.
        """
        self.log('[DATASET] Loading reviews...')
        reviews = []
        for inter in self._parse_gz(path):
            user = inter['reviewerID']
            item = inter['asin']
            time = inter['unixReviewTime']
            reviews.append((user, item, int(time)))
        return reviews

    def _get_item_seqs(self, reviews: list[tuple]) -> dict:
        """
        Group the reviews by user and sort the items by time.

        Args:
            reviews (list[tuple]): A list of tuples representing the reviews. Each tuple contains the user, item, and time.

        Returns:
            dict: A dictionary where the keys are the users and the values are lists of items sorted by time.
        """
        # Group reviews by user
        item_seqs = defaultdict(list)
        for data in reviews:
            user, item, time = data
            item_seqs[user].append((item, time))

        # Sort items by time
        for user, item_time in item_seqs.items():
            item_time.sort(key=lambda x: x[1])
            item_seqs[user] = [_[0] for _ in item_time]
        return item_seqs

    def _remap_ids(self, item_seqs: dict):
        """
        Remaps the user and item IDs in the given item sequences dictionary.

        Args:
            item_seqs (dict): A dictionary containing user-item sequences, where the keys are the users and the values are lists of items sorted by time.

        Returns:
            all_item_seqs (dict): A dictionary containing the user-item sequences.
            id_mapping (dict): A dictionary containing the mapping between raw and remapped user and item IDs.
                - user2id (dict): A dictionary mapping raw user IDs to remapped user IDs.
                - item2id (dict): A dictionary mapping raw item IDs to remapped item IDs.
                - id2user (list): A list mapping remapped user IDs to raw user IDs.
                - id2item (list): A list mapping remapped item IDs to raw item IDs.

        Note:
            The remapped user and item IDs start from 1. The ID 0 is reserved for padding `[PAD]`.
        """
        self.log('[DATASET] Remapping user and item IDs...')
        for user, items in item_seqs.items():
            if user not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
                self.id_mapping['id2user'].append(user)
            iids = []           # item id lists
            for item in items:
                if item not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
                    self.id_mapping['id2item'].append(item)
                iids.append(item)
            self.all_item_seqs[user] = iids
        return self.all_item_seqs, self.id_mapping

    def _process_reviews(self,
        output_path: str
    ) -> tuple[dict, dict]:
        """
        Process the reviews from the input path and save the data to the output path.

        Args:
            output_path (str): The path to save the data.

        Returns:
            all_item_seqs (dict): A dictionary containing the user-item sequences.
            id_mapping (dict): A dictionary containing data maps.
        """
        # Check if the processed data already exists
        os.makedirs(os.path.join(output_path, 'processed'), exist_ok=True)
        seq_file = os.path.join(output_path, 'processed', 'all_item_seqs.json')
        id_mapping_file = os.path.join(output_path, 'processed', 'id_mapping.json')
        if os.path.exists(seq_file) and os.path.exists(id_mapping_file):
            self.log('[DATASET] Reviews have been processed...')
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            return all_item_seqs, id_mapping

        self.log('[DATASET] Processing reviews...')

        # Load reviews
        with open(os.path.join(output_path, 'raw', 'raw_item_seqs.json'), 'r') as f:
            raw_item_seqs = json.load(f)
        all_item_seqs, id_mapping = self._remap_ids(raw_item_seqs)

        # Save data
        self.log('[DATASET] Saving mapping data...')
        with open(seq_file, 'w') as f:
            json.dump(all_item_seqs, f)
        with open(id_mapping_file, 'w') as f:
            json.dump(id_mapping, f)
        return all_item_seqs, id_mapping

    def _rand_init_meta(
        self,
        item2id: dict
    ) -> dict:
        item2meta = {}
        n_digit = self.config['rq_n_codebooks'] + 1
        n_codebook_size = self.config['rq_codebook_size']
        existing_sids = set()
        for item in tqdm(item2id.keys(), desc='[DATASET] Initializing metadata'):
            sid = np.random.randint(0, n_codebook_size, size=(n_digit,)).tolist()
            sid_str = ','.join([str(s) for s in sid])
            while sid_str in existing_sids:
                sid = np.random.randint(0, n_codebook_size, size=(n_digit,)).tolist()
                sid_str = ','.join([str(s) for s in sid])
            existing_sids.add(sid_str)
            item2meta[item] = sid
        return item2meta

    def _process_meta(
        self,
        output_path: str
    ) -> Optional[dict]:
        """
        Process metadata based on the specified process type.

        Args:
            input_path (str): The path to the input metadata file.
            output_path (str): The path to save the processed metadata file.

        Returns:
            dict: A dictionary containing the item metadata.

        Raises:
            NotImplementedError: If the metadata processing type is not implemented.
        """
        if 'sent_emb_model' not in self.config:
            self.log('[DATASET] No metadata processing required...')
            return {}
        base_model = self.config['sent_emb_model'].split('/')[-1]
        n_digit = self.config['rq_n_codebooks'] + 1
        n_codebook_size = self.config['rq_codebook_size']
        sid_str = ",".join(map(str, [n_codebook_size] * n_digit))
        meta_file = os.path.join(output_path, 'processed', f'{base_model}_{sid_str}.sem_ids')
        if os.path.exists(meta_file):
            self.log('[DATASET] Metadata has been processed...')
            with open(meta_file, 'r') as f:
                return json.load(f)

        item2meta = self._rand_init_meta(item2id=self.item2id)

        with open(meta_file, 'w') as f:
            json.dump(item2meta, f)
        return item2meta

    def _download_and_process_raw(self):
        """
        Downloads and processes the raw data files.

        This method downloads the raw data files for reviews and metadata from the specified path,
        processes the raw data, and saves the processed data in the cache directory.

        Returns:
            None
        """

        # Following https://github.com/RUCAIBox/CIKM2020-S3Rec/blob/master/data/data_process.py
        np.random.seed(12345)

        # Process raw data
        os.makedirs(self.cache_dir, exist_ok=True)

        self.all_item_seqs, self.id_mapping = self._process_reviews(
            output_path=self.cache_dir
        )

        self.item2meta = self._process_meta(
            output_path=self.cache_dir
        )
