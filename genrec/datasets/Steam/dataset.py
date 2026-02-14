import collections
import os
import json
import gzip
import shutil
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text


class Steam(AbstractDataset):
    """
    Steam dataset class for handling Steam game reviews and metadata.
    
    This class downloads and processes the Steam dataset for recommendation tasks.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the Steam dataset.
        
        Args:
            config (dict): Configuration dictionary containing dataset parameters
        """
        super(Steam, self).__init__(config)
        
        self.log(f'[DATASET] Steam Dataset')
        
        self.cache_dir = os.path.join(config['cache_dir'], 'Steam')
        self.log(f'[DATASET] Cache directory: {self.cache_dir}')
        self._download_and_process_raw()

    def _download_steam_dataset(self, raw_data_dir: str) -> None:
        """
        Download Steam dataset files.
        
        Args:
            raw_data_dir (str): Directory to save the downloaded files
        """
        self.log(f'[DATASET] Downloading Steam dataset to {raw_data_dir}')
        os.makedirs(raw_data_dir, exist_ok=True)
        
        # URLs of files to download
        urls = [
            "https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
            "https://cseweb.ucsd.edu/~wckang/steam_games.json.gz"
        ]
        
        for url in urls:
            # Get filename from URL
            filename = os.path.basename(url)
            output_path = os.path.join(raw_data_dir, filename)
            
            # Download file if it doesn't exist
            if not os.path.exists(output_path):
                self.log(f"[DATASET] Downloading {filename}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                with open(output_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
            else:
                self.log(f"[DATASET] {filename} already exists, skipping download")
            
        self.log("[DATASET] Download completed")
    
    def _parse(self, path: str):
        """
        Parse a gzipped file containing Python literal dictionaries.
        
        Args:
            path (str): Path to the gzipped file
            
        Yields:
            dict: Parsed dictionary from each line
        """
        with gzip.open(path, 'rt', encoding='utf-8') as g:
            for line in g:
                try:
                    yield eval(line)
                except Exception as e:
                    self.log(f"[DATASET] Error parsing line: {e}")
                    continue
    
    def _load_steam_review(self, raw_data_dir: str) -> List[Tuple[str, str, int]]:
        """
        Load Steam reviews from the downloaded file.
        
        Args:
            raw_data_dir (str): Directory containing the raw data files
            
        Returns:
            List[Tuple[str, str, int]]: List of (user, item, timestamp) tuples
        """
        self.log(f"[DATASET] Loading Steam reviews...")
        datas = []
        file_path = os.path.join(raw_data_dir, "steam_reviews.json.gz")
        
        for review in tqdm(self._parse(file_path), desc="Parsing reviews"):
            user = review['username']
            item = review['product_id']
            # Convert date string to Unix timestamp
            date_str = review['date']
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                timestamp = int(time.mktime(dt.timetuple()))
                datas.append((user, item, timestamp))
            except:
                continue
                
        self.log(f"[DATASET] Loaded {len(datas)} reviews")
        return datas
    
    def _load_steam_metadata(self, raw_data_dir: str, data_maps: Dict) -> Dict[str, Any]:
        """
        Load Steam game metadata.
        
        Args:
            raw_data_dir (str): Directory containing the raw data files
            data_maps (Dict): Dictionary containing item ID mappings
            
        Returns:
            Dict[str, Any]: Dictionary mapping item IDs to metadata
        """
        self.log(f"[DATASET] Loading Steam metadata...")
        meta_infos = {}
        meta_file = os.path.join(raw_data_dir, "steam_games.json.gz")
        item_ids = list(data_maps['item2id'].keys())

        for info in tqdm(self._parse(meta_file), desc="Parsing metadata"):
            if 'id' not in info or info['id'] not in item_ids:
                continue
            meta_infos[info['id']] = info

        self.log(f"[DATASET] {len(meta_infos)} out of {len(data_maps['item2id']) - 1} items have metadata")
        return meta_infos

    def _get_interaction(self, datas: List[Tuple[str, str, int]]) -> Dict[str, List[str]]:
        """
        Get user-item interactions sorted by timestamp.
        
        Args:
            datas (List[Tuple[str, str, int]]): List of (user, item, timestamp) tuples
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping users to their item sequences
        """
        self.log(f"[DATASET] Creating user-item interaction sequences...")
        user_seq = {}
        for data in datas:
            user, item, time = data
            if user in user_seq:
                user_seq[user].append((item, time))
            else:
                user_seq[user] = []
                user_seq[user].append((item, time))

        # Sort interactions by timestamp and extract items
        for user, item_time in user_seq.items():
            item_time.sort(key=lambda x: x[1])  # Sort by timestamp
            items = []
            for t in item_time:
                items.append(t[0])
            user_seq[user] = items
            
        return user_seq

    def _check_Kcore(self, user_items: Dict[str, List[str]], user_core: int, item_core: int) -> Tuple[Dict[str, int], Dict[str, int], bool]:
        """
        Check if the dataset satisfies K-core filtering requirements.
        
        Args:
            user_items: Dictionary mapping users to their item lists
            user_core: Minimum number of items per user
            item_core: Minimum number of users per item
            
        Returns:
            Tuple containing user counts, item counts, and K-core status
        """
        user_count = defaultdict(int)
        item_count = defaultdict(int)
        for user, items in user_items.items():
            for item in items:
                user_count[user] += 1
                item_count[item] += 1

        for user, num in user_count.items():
            if num < user_core:
                return user_count, item_count, False
                
        for item, num in item_count.items():
            if num < item_core:
                return user_count, item_count, False
                
        return user_count, item_count, True  # K-core requirements satisfied

    def _filter_Kcore(self, user_items: Dict[str, List[str]], user_core: int, item_core: int) -> Dict[str, List[str]]:
        """
        Filter dataset to satisfy K-core requirements.
        
        Args:
            user_items: Dictionary mapping users to their item lists
            user_core: Minimum number of items per user
            item_core: Minimum number of users per item
            
        Returns:
            Filtered user_items dictionary
        """
        self.log(f"[DATASET] Filtering dataset to {user_core}-core for users, {item_core}-core for items...")
        user_count, item_count, isKcore = self._check_Kcore(user_items, user_core, item_core)
        
        iteration = 0
        while not isKcore:
            iteration += 1
            self.log(f"[DATASET] K-core filtering iteration {iteration}")
            for user, num in user_count.items():
                if user_count[user] < user_core: # delete user
                    user_items.pop(user)
                else:
                    for item in user_items[user]:
                        if item_count[item] < item_core:
                            user_items[user].remove(item)
            user_count, item_count, isKcore = self._check_Kcore(user_items, user_core, item_core)
        return user_items
    
    def _subsample_user_items(self, user_items: dict, n: int) -> dict:
        """
        Subsamples every nth user from the user_items dict.

        Args:
            user_items (dict): Mapping of user_id to item sequence.
            n (int): Subsampling factor. Keep every nth user.

        Returns:
            dict: Subsampled user_items dictionary.
        """
        sampled_user_items = {
            user_id: items
            for idx, (user_id, items) in enumerate(user_items.items())
            if idx % n == 0
        }
        return sampled_user_items

    def _id_map(self, user_items: Dict[str, List[str]]) -> Tuple[Dict[int, List[int]], int, int, Dict]:
        """
        Map original user and item IDs to sequential integer IDs.
        
        Args:
            user_items: Dictionary mapping original user IDs to lists of original item IDs
            
        Returns:
            Tuple containing:
            - Dictionary mapping new user IDs to lists of new item IDs
            - Number of users
            - Number of items
            - ID mapping dictionaries
        """
        self.log(f"[DATASET] Mapping IDs to sequential integers...")
        
        # Use the class id_mapping
        user2id = self.id_mapping['user2id']
        item2id = self.id_mapping['item2id']
        id2user = self.id_mapping['id2user']
        id2item = self.id_mapping['id2item']

        final_data = {}
        for user, items in user_items.items():
            if user not in user2id:
                user2id[user] = len(user2id)
                id2user.append(user)
            
            iids = []  # item id lists
            for item in items:
                if item not in item2id:
                    item2id[item] = len(item2id)
                    id2item.append(item)
                iids.append(item2id[item])
            
            uid = user2id[user]
            final_data[uid] = iids
        
        # Count unique users and items (excluding padding token)
        num_users = len(user2id) - 1
        num_items = len(item2id) - 1
        
        self.log(f"[DATASET] Mapped {num_users} users and {num_items} items")
        return final_data, num_users, num_items, self.id_mapping

    def _format_list(self, items: List[str]) -> str:
        """Format a list of items into a readable string."""
        if not items:
            return ""
        
        # Remove any empty strings
        items = [item for item in items if item]
        
        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _feature_process(self, feature_name, feature_value):
        """Process a single feature into a sentence format."""
        if feature_value is None or feature_value == "":
            return ""
        
        # Define mapping of feature names to formatting functions
        formatters = {
            # Name features
            "title": lambda val: f"Name: {clean_text(val)}.",
            "app_name": lambda val: f"Name: {clean_text(val)}.",
            
            # List features
            "genres": lambda val: f"Genres: {self._format_list(val)}." if isinstance(val, list) and val else "",
            "tags": lambda val: f"Tags: {self._format_list(val)}." if isinstance(val, list) and val else "",
            "specs": lambda val: f"Features: {self._format_list(val)}." if isinstance(val, list) and val else "",
            
            # Rating features
            "sentiment": lambda val: f"Reviews: {clean_text(val)}.",
            "metascore": lambda val: f"Metascore: {val}/100.",
            
            # Price features
            "price": lambda val: "Price: Free to Play." if str(val).lower() in ["free to play", "free to play"] 
                                else f"Price: ${val}.",
            "discount_price": lambda val: f"Discount price: ${val}.",
            
            # Metadata features
            "release_date": lambda val: f"Released: {clean_text(val)}.",
            "developer": lambda val: f"Developer: {clean_text(val)}.",
            "publisher": lambda val: f"Publisher: {clean_text(val)}.",
            
            # Boolean features
            "early_access": lambda val: "In Early Access." if val else "",
        }
        
        # Get the appropriate formatter and apply it, or apply default cleaning if no formatter exists
        formatter = formatters.get(feature_name)
        if formatter:
            return formatter(feature_value)
        else:
            # Default handling for any other feature: clean the text and add feature name
            return f"{feature_name.replace('_', ' ').title()}: {clean_text(feature_value)}."
    
    def _clean_metadata(self, game_data: Dict[str, Any]) -> str:
        """
        Process a game metadata entry into a string of descriptive sentences.
        
        Args:
            game_data: Dictionary containing game metadata
            
        Returns:
            String of sentences describing the game
        """
        meta_text = ""
        
        # Define important features to include in a specific order
        features_needed = [
            'title', 'genres', 'tags', 'sentiment', 'metascore',
            'price', 'discount_price', 'release_date', 
            'developer', 'publisher', 'specs', 'early_access'
        ]
        
        for feature in features_needed:
            if feature in game_data and game_data[feature] is not None:
                feature_text = self._feature_process(feature, game_data[feature])
                if feature_text:
                    meta_text += feature_text + " "
        
        return meta_text.strip()

    def _extract_meta_sentences(self, game_map: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract metadata sentences from a dictionary of game data.
        
        Args:
            game_map: Dictionary mapping game_id to game metadata
            
        Returns:
            Dictionary mapping game_id to metadata sentences
        """
        self.log(f"[DATASET] Extracting metadata sentences for {len(game_map)} games...")
        item2meta = {}
        
        for game_id, game_data in game_map.items():
            item2meta[game_id] = self._clean_metadata(game_data)
        
        return item2meta

    def _process_meta(self, game_map: Dict[str, Dict[str, Any]], output_dir: str) -> Optional[Dict[str, Any]]:
        """
        Process metadata from a game map and save to file.
        
        Args:
            game_map: Dictionary mapping game_id to game metadata
            output_dir: Directory to save the processed metadata
            
        Returns:
            Processed metadata dictionary or None if mode is 'none'
        """
        process_mode = self.config.get('metadata', 'sentence')
        self.log(f'[DATASET] Processing metadata, mode: {process_mode}')
        
        metadata_file = os.path.join(output_dir, f"metadata.{process_mode}.json")
        if os.path.exists(metadata_file):
            self.log(f'[DATASET] Loading processed metadata from {metadata_file}')
            with open(metadata_file, 'r') as f:
                return json.load(f)
        
        if process_mode == 'none':
            # No metadata processing required
            return None
            
        if process_mode == 'sentence':
            item2meta = self._extract_meta_sentences(game_map)
        else:
            raise NotImplementedError(f'Metadata processing mode "{process_mode}" not implemented.')
        
        os.makedirs(output_dir, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(item2meta, f)
        
        return item2meta

    def _timestamp_split(self, user_items: Dict[str, List[str]]) -> Dict[str, Dataset]:
        """
        Split the dataset based on timestamps.
        
        Args:
            user_items: Dictionary mapping users to their item sequences
            
        Returns:
            Dictionary containing the split datasets
        """
        self.log(f"[DATASET] Splitting dataset by timestamp...")
        raise NotImplementedError('Split by timestamp not implemented yet.')

    def _download_and_process_raw(self):
        """
        Download and process the raw Steam dataset.
        
        This method:
        1. Downloads the Steam reviews and games datasets
        2. Filters the data based on K-core
        3. Maps IDs to sequential integers
        4. Processes metadata
        5. Prepares the final dataset for training
        """
        user_core = self.config.get('user_core', 5)
        item_core = self.config.get('item_core', 5)
        subsample_every_n_users = self.config.get('subsample_every_n_users', 7)

        processed_data_path = os.path.join(self.cache_dir, 'processed')
        if os.path.exists(processed_data_path):
            self.log(f'[DATASET] Loading processed data from {processed_data_path}')
            
            # Load ID mappings
            id_mapping_file = os.path.join(processed_data_path, 'id_mapping.json')
            if os.path.exists(id_mapping_file):
                with open(id_mapping_file, 'r') as f:
                    self.id_mapping = json.load(f)
            
            # Load item sequences
            seq_file = os.path.join(processed_data_path, 'all_item_seqs.json')
            if os.path.exists(seq_file):
                with open(seq_file, 'r') as f:
                    self.all_item_seqs = json.load(f)
                    
            # Load metadata
            meta_file = os.path.join(processed_data_path, f"metadata.{self.config.get('metadata', 'sentence')}.json")
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    self.item2meta = json.load(f)
                    
            return
        
        # Download and process raw data
        raw_data_dir = os.path.join(self.cache_dir, "raw")
        self._download_steam_dataset(raw_data_dir)
        
        self.log(f'[DATASET] Loading raw data from {raw_data_dir}')
        review_data = self._load_steam_review(raw_data_dir)
        
        # Get initial user-item interactions
        user_items = self._get_interaction(review_data)
        self.log(f'[DATASET] Found {len(user_items)} users with {sum(len(items) for items in user_items.values())} interactions')

        # Get initial statistics before filtering
        initial_user_count, initial_item_count, initial_kcore_satisfied = self._check_Kcore(user_items, user_core=user_core, item_core=item_core)
        self.log(f'[DATASET] Initial dataset has {len(initial_user_count)} users and {len(initial_item_count)} items')
        self.log(f'[DATASET] Initial K-core satisfied: {initial_kcore_satisfied}')

        # Filter with K-core
        user_items = self._filter_Kcore(user_items, user_core=user_core, item_core=item_core)
        self.log(f'[DATASET] After filtering: {len(user_items)} users')
        
        # Verify K-core requirements are satisfied after filtering
        user_count, item_count, kcore_satisfied = self._check_Kcore(user_items, user_core=user_core, item_core=item_core)
        assert kcore_satisfied, "K-core requirements should be satisfied after filtering"
        
        # Subsample users to reduce dataset size
        sampled_user_items = self._subsample_user_items(user_items, subsample_every_n_users)
        self.log(f'[DATASET] After subsampling: {len(sampled_user_items)} users')

        # Recalculate user and item counts
        user_count, item_count, _ = self._check_Kcore(sampled_user_items, user_core=user_core, item_core=item_core)
        self.log(f'[DATASET] Final dataset has {len(user_count)} unique users and {len(item_count)} unique items')

        # Remap IDs
        user_items_mapped, user_num, item_num, data_maps = self._id_map(sampled_user_items)

        # Compute and log statistics
        user_count_list = list(user_count.values())
        user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
        
        item_count_list = list(item_count.values())
        item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
        
        interact_num = np.sum([x for x in user_count_list])
        sparsity = (1 - interact_num / (user_num * item_num)) * 100
        
        stats_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                    f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                    f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
        self.log(f'[DATASET] Dataset statistics:\n{stats_info}')
        
        # Load metadata
        meta_data = self._load_steam_metadata(raw_data_dir, data_maps)
        
        # Create processed directory
        os.makedirs(processed_data_path, exist_ok=True)
        
        # Save ID mappings
        with open(os.path.join(processed_data_path, 'id_mapping.json'), 'w') as f:
            json.dump(data_maps, f)
        
        # Process metadata
        self.item2meta = self._process_meta(meta_data, processed_data_path)
        
        # Map back to original IDs for compatibility with the whole codebase
        original_user_items = {}
        for uid, iids in user_items_mapped.items():
            original_user_items[data_maps['id2user'][uid]] = [data_maps['id2item'][i] for i in iids]
        
        self.all_item_seqs = original_user_items
        
        # Save item sequences
        with open(os.path.join(processed_data_path, 'all_item_seqs.json'), 'w') as f:
            json.dump(self.all_item_seqs, f)
        
        # Handle timestamp-based splitting if needed
        if self.config.get('split') == 'timestamp':
            self.split_data = self._timestamp_split(self.all_item_seqs)