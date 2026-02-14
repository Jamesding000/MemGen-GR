import collections
import os
import json
from datasets import load_dataset, Dataset

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text


class AmazonReviews2023(AbstractDataset):
    def __init__(self, config: dict):
        super(AmazonReviews2023, self).__init__(config)

        self.category = config['category']
        self._check_available_category()
        self.log(
            f'[DATASET] Amazon Reviews 2023 for category: {self.category}'
        )

        self.cache_dir = os.path.join(
            config['cache_dir'], 'AmazonReviews2023', self.category
        )
        self._download_and_process_raw()

    def _check_available_category(self):
        """
        Checks if the `self.category` is available in the dataset.

        Raises:
            AssertionError: If the specified category is not available.
        """
        available_categories = [
            'All_Beauty',
            'Amazon_Fashion',
            'Appliances',
            'Arts_Crafts_and_Sewing',
            'Automotive',
            'Baby_Products',
            'Beauty_and_Personal_Care',
            'Books',
            'CDs_and_Vinyl',
            'Cell_Phones_and_Accessories',
            'Clothing_Shoes_and_Jewelry',
            'Digital_Music',
            'Electronics',
            'Gift_Cards',
            'Grocery_and_Gourmet_Food',
            'Handmade_Products',
            'Health_and_Household',
            'Health_and_Personal_Care',
            'Home_and_Kitchen',
            'Industrial_and_Scientific',
            'Kindle_Store',
            'Magazine_Subscriptions',
            'Movies_and_TV',
            'Musical_Instruments',
            'Office_Products',
            'Patio_Lawn_and_Garden',
            'Pet_Supplies',
            'Software',
            'Sports_and_Outdoors',
            'Subscription_Boxes',
            'Tools_and_Home_Improvement',
            'Toys_and_Games',
            'Unknown',
            'Video_Games',
        ]
        assert self.category in available_categories, \
            f'Category "{self.category}" not available. ' \
            f'Available categories: {available_categories}'

        if self.config['kcore'] == '5core':
            if self.category in [
                'Amazon_Fashion',
                'Appliances',
                'Digital_Music',
                'Handmade_Products',
                'Health_and_Personal_Care',
                'Subscription_Boxes',
            ]:
                raise ValueError(
                    f'[DATASET] Category "{self.category}" does not have 5-core '
                    f'reviews. Using 5-core reviews for other categories.'
                )

    def _remap_ids(self, datasets, output_path: str):
        id_mapping_file = os.path.join(output_path, 'id_mapping.json')
        if os.path.exists(id_mapping_file):
            self.log(f'[DATASET] Loading id mapping from {id_mapping_file}')
            id_mapping = json.load(open(id_mapping_file, 'r'))
            return id_mapping

        for split in ['train', 'valid', 'test']:
            dataset = datasets[split]
            for user_id, item_id, history in zip(
                dataset['user_id'],
                dataset['parent_asin'],
                dataset['history'],
            ):
                if user_id not in self.id_mapping['user2id']:
                    self.id_mapping['user2id'][user_id] = len(self.id_mapping['user2id'])
                    self.id_mapping['id2user'].append(user_id)
                if item_id not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item_id] = len(self.id_mapping['item2id'])
                    self.id_mapping['id2item'].append(item_id)
                items_in_history = history.split(' ')
                for item in items_in_history:
                    if item not in self.id_mapping['item2id']:
                        self.id_mapping['item2id'][item] = len(self.id_mapping['item2id'])
                        self.id_mapping['id2item'].append(item)

        with open(id_mapping_file, 'w') as f:
            json.dump(self.id_mapping, f)
        return self.id_mapping

    def _feature_process(self, feature):
        sentence = ""
        if isinstance(feature, float):
            sentence += str(feature)
            sentence += '.'
        elif isinstance(feature, list) and len(feature) > 0:
            for v in feature:
                sentence += clean_text(v)
                sentence += ', '
            sentence = sentence[:-2]
            sentence += '.'
        else:
            sentence = clean_text(feature)
        return sentence + ' '

    def _clean_metadata(self, example, concat=True):

        features_needed = ['title', 'features', 'categories', 'description']
        if concat:
            meta_text = ''
            for feature in features_needed:
                meta_text += self._feature_process(example[feature])
            example['cleaned_metadata'] = meta_text
        else:
            for feature in features_needed:
                meta_text = self._feature_process(example[feature])
                example[feature] = meta_text.strip()

        return example

    def _extract_meta_sentences(self, meta_dataset):
        meta_dataset = meta_dataset.map(
            lambda t: self._clean_metadata(t),
            num_proc=self.config['num_proc']
        )

        item2meta = {}
        for parent_asin, cleaned_metadata in zip(
            meta_dataset['parent_asin'],
            meta_dataset['cleaned_metadata']
        ):
            item2meta[parent_asin] = cleaned_metadata
        return item2meta


    def _extract_meta_sentence_features(self, meta_dataset):

        features_needed = ['title', 'features', 'categories', 'description']

        meta_dataset = meta_dataset.map(
            lambda t: self._clean_metadata(t, concat=False),
            num_proc=self.config['num_proc']
        )

        item2meta = collections.defaultdict(dict)
        for example in meta_dataset:
            parent_asin = example['parent_asin']
            for feature in features_needed:
                item2meta[parent_asin][feature] = example[feature]

        return item2meta

    def _process_meta(self, output_path: str):
        process_mode = self.config['metadata']
        meta_file = os.path.join(output_path, f'metadata.{process_mode}.json')
        if os.path.exists(meta_file):
            self.log(f'[DATASET] Metadata has been processed...')
            item2meta = json.load(open(meta_file, 'r'))
            return item2meta

        self.log(f'[DATASET] Processing metadata, mode: {process_mode}')

        if process_mode == 'none':
            # No metadata processing required
            return None

        meta_dataset = load_dataset(
            'McAuley-Lab/Amazon-Reviews-2023',
            f'raw_meta_{self.category}',
            split='full',
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        meta_dataset = meta_dataset.filter(
            lambda t: t['parent_asin'] in self.id_mapping['item2id']
        )
        self.log(
            f'[DATASET] {len(meta_dataset)} of '
            f'{len(self.id_mapping["item2id"]) - 1} items have meta data.'
        )

        if process_mode == 'sentence':
            item2meta = self._extract_meta_sentences(meta_dataset=meta_dataset)
        elif process_mode == 'sentence_feature':
            item2meta = self._extract_meta_sentence_features(meta_dataset=meta_dataset)
        else:
            raise NotImplementedError(f'Metadata processing mode "{process_mode}" not implemented.')

        with open(meta_file, 'w') as f:
            json.dump(item2meta, f)
        return item2meta

    def _merge_augmented_dataset(self, datasets, output_path: str):
        seq_file = os.path.join(output_path, 'all_item_seqs.json')
        if os.path.exists(seq_file):
            self.log(f'[DATASET] Loading merged dataset from {seq_file}')
            all_item_seqs = json.load(open(seq_file, 'r'))
            return all_item_seqs

        for split in ['train', 'valid', 'test']:
            for user, parent_asin, history in zip(
                datasets[split]['user_id'],
                datasets[split]['parent_asin'],
                datasets[split]['history'],
            ):
                items_in_history = history.split(' ')
                items_in_history.append(parent_asin)
                if user not in self.all_item_seqs:
                    self.all_item_seqs[user] = []
                if len(items_in_history) > len(self.all_item_seqs[user]):
                    self.all_item_seqs[user] = items_in_history

        with open(seq_file, 'w') as f:
            json.dump(self.all_item_seqs, f)
        return self.all_item_seqs

    def _filter_non_history(self, datasets):
        """
        Filters out examples from the datasets that have an empty 'history' field.

        Args:
            datasets (dict): A dictionary containing the datasets for 'train', 'valid', and 'test' splits.

        Returns:
            dict: A dictionary containing the filtered datasets for 'train', 'valid', and 'test' splits.
        """
        for split in ['train', 'valid', 'test']:
            datasets[split] = datasets[split].filter(
                lambda t: len(t['history']) > 0
            )
        return datasets

    def _timestamp_split(self, datasets):
        split_data = {}
        for split in ['train', 'valid', 'test']:
            split_data[split] = {'user': [], 'item_seq': []}
            for user, parent_asin, history in zip(
                datasets[split]['user_id'],
                datasets[split]['parent_asin'],
                datasets[split]['history'],
            ):
                items_in_history = history.split(' ')
                items_in_history.append(parent_asin)
                split_data[split]['user'].append(user)
                split_data[split]['item_seq'].append(items_in_history)
        return {
            'val' if k == 'valid' else k: Dataset.from_dict(v)
            for k, v in split_data.items()
        }

    def _download_and_process_raw(self):
        """
        Downloads and processes the raw data for the AmazonReviews2023 dataset.

        This method performs the following steps:
        1. Downloads the processed reviews dataset from the McAuley-Lab/Amazon-Reviews-2023 repository.
        2. Filters out non-history datasets.
        3. Creates a directory for storing the processed data.
        4. Remaps the IDs in the datasets.
        5. Downloads and processes the metadata.
        6. Merges the augmented dataset to fit the whole codebase.
        """

        processed_data_path = os.path.join(self.cache_dir, 'processed')
        if os.path.exists(processed_data_path):
            datasets = None
        else:
            raw_data_dir = os.path.join(
                self.cache_dir,
                f"raw/benchmark/{self.config['kcore']}/{self.config['split']}_w_his"
            )
            if os.path.isdir(raw_data_dir):
                self.logger.info(f'[DATASET] Loading raw data from {raw_data_dir}')
                datasets = load_dataset(
                    'csv',
                    data_files={
                        split: os.path.join(raw_data_dir, f'{self.category}.{split}.csv')
                        for split in ['train', 'valid', 'test']
                    }
                )
            else:
                # Download processed reviews
                with self.accelerator.main_process_first(): # only download once when ddp
                    datasets = load_dataset(
                        "McAuley-Lab/Amazon-Reviews-2023",
                        f"{self.config['kcore']}_{self.config['split']}_w_his_{self.category}",
                        cache_dir=self.cache_dir,
                        trust_remote_code=True,
                    )

            datasets = self._filter_non_history(datasets)

            processed_data_path = os.path.join(self.cache_dir, 'processed')
            os.makedirs(processed_data_path, exist_ok=True)

        self.id_mapping = self._remap_ids(
            datasets=datasets,
            output_path=processed_data_path
        )

        # Download and process metadata
        self.item2meta = self._process_meta(
            output_path=processed_data_path
        )

        # The original benchmark has been augmented
        # Merge to fit the whole codebase
        self.all_item_seqs = self._merge_augmented_dataset(
            datasets=datasets,
            output_path=processed_data_path
        )

        if self.config['split'] == 'timestamp':
            self.split_data = self._timestamp_split(datasets)
