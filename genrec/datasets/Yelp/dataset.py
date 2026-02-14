import collections
import os
import json
import shutil
import time
import html
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset
import kagglehub

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text


class Yelp(AbstractDataset):
    """
    Yelp dataset class for handling Yelp business reviews and metadata.
    
    This class downloads and processes the Yelp dataset for recommendation tasks.
    """

    def __init__(self, config: dict):
        """
        Initialize the Yelp dataset.
        
        Args:
            config (dict): Configuration dictionary containing dataset parameters
        """
        super(Yelp, self).__init__(config)
        
        self.version = config['version']
        self._check_available_version()
        self.log(f'[DATASET] Yelp Dataset (Version: {self.version})')
        
        self.cache_dir = os.path.join(config['cache_dir'], 'Yelp', self.version)
        self.log(f'[DATASET] Cache directory: {self.cache_dir}')
        self._download_and_process_raw()
    
    def _check_available_version(self):
        """
        Checks if the `self.version` is available in the dataset.

        Raises:
            AssertionError: If the specified version is not available.
        """
        available_versions = [
            "Yelp_2018",
            "Yelp_2020",
            "Yelp_2021",
            "Yelp_2022",
        ]
        assert self.version in available_versions, \
            f'Version "{self.version}" not available. ' \
            f'Available verions: {available_versions}'

    def _download_yelp_dataset(self, raw_data_dir: str) -> None:
        """
        Download Yelp dataset files from Kaggle.
        
        Args:
            raw_data_dir (str): Directory to save the downloaded files
        """
        self.log(f'[DATASET] Downloading Yelp dataset to {raw_data_dir}')
        
        VERSION_PATH = {
            "Yelp_2018": "yelp-dataset/yelp-dataset/versions/1",
            "Yelp_2020": "yelp-dataset/yelp-dataset/versions/2",
            "Yelp_2021": "yelp-dataset/yelp-dataset/versions/3",
            "Yelp_2022": "yelp-dataset/yelp-dataset",
        }

        if os.path.exists(raw_data_dir) \
            and os.path.exists(os.path.join(raw_data_dir, "yelp_academic_dataset_review.json")) \
            and os.path.exists(os.path.join(raw_data_dir, "yelp_academic_dataset_business.json")):
                
            self.log(f"[DATASET] Yelp dataset already exists, skipping download")
            return

        # Download raw dataset from Kaggle
        try:
            # Download the dataset to a temporary directory
            temp_path = kagglehub.dataset_download(VERSION_PATH[self.version])
            self.log(f"[DATASET] Kaggle dataset downloaded to temp_path: {temp_path}")

            # Move downloaded files to the target directory
            os.makedirs(raw_data_dir, exist_ok=True)

            for item in os.listdir(temp_path):
                shutil.move(os.path.join(temp_path, item), raw_data_dir)

            # Remove the temporary directory
            shutil.rmtree(temp_path)

            self.log(f"[DATASET] Downloaded raw Yelp dataset to: {raw_data_dir}")

        except Exception as e:
            self.log(f"[DATASET] Error downloading Yelp dataset: {e}")
            raise

    def _load_yelp_review(self, raw_data_dir: str, date_min: Optional[str] = None, date_max: Optional[str] = None, rating_score: float = 0.0) -> List[Tuple[str, str, int]]:
        """
        Load Yelp reviews from the downloaded file.
        
        Args:
            raw_data_dir (str): Directory containing the raw data files
            date_min (str, optional): Minimum date for filtering reviews, None to skip minimum date filtering
            date_max (str, optional): Maximum date for filtering reviews, None to skip maximum date filtering
            rating_score (float): Minimum rating score for filtering reviews
            
        Returns:
            List[Tuple[str, str, int]]: List of (user, item, timestamp) tuples
        """
        self.log(f"[DATASET] Loading Yelp reviews...")
        datas = []
        data_file = os.path.join(raw_data_dir, "yelp_academic_dataset_review.json")

        with open(data_file, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Processing reviews"):
            review = json.loads(line.strip())
            user = review['user_id']
            item = review['business_id']
            rating = review['stars']
            date = review['date']

            # Filter reviews based on date and rating
            if ((date_min is not None and date < date_min) or 
                (date_max is not None and date > date_max) or 
                float(rating) <= rating_score):
                continue

            # Convert date to timestamp format
            time_str = date.replace('-', '').replace(':', '').replace(' ', '')
            datas.append((user, item, int(time_str)))

        self.log(f"[DATASET] Loaded {len(datas)} reviews after filtering")
        return datas

    def _load_yelp_metadata(self, raw_data_dir: str, data_maps: Dict) -> Dict[str, Any]:
        """
        Load Yelp business metadata.
        
        Args:
            raw_data_dir (str): Directory containing the raw data files
            data_maps (Dict): Dictionary containing item ID mappings
            
        Returns:
            Dict[str, Any]: Dictionary mapping business IDs to metadata
        """
        self.log(f"[DATASET] Loading Yelp business metadata...")
        meta_infos = {}
        meta_file = os.path.join(raw_data_dir, "yelp_academic_dataset_business.json")
        item_ids = list(data_maps['item2id'].keys())

        with open(meta_file, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Processing business metadata"):
            info = json.loads(line)
            if info['business_id'] not in item_ids:
                continue
            meta_infos[info['business_id']] = info

        self.log(f"[DATASET] {len(meta_infos)} out of {len(data_maps['item2id']) - 1} businesses have metadata")
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

        # Remove any empty strings and clean each item
        items = [clean_text(item) for item in items if item]

        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _process_attributes(self, attributes: Dict) -> List[str]:
        """Process a business attributes dictionary into sentences."""
        sentences = []

        # Group similar attributes to avoid repetition
        grouped_attributes = defaultdict(list)

        for attr_name, attr_value in attributes.items():
            # Skip False values
            if attr_value == "False" or attr_value is False or attr_value == "'False'":
                continue
            
            clean_value = clean_text(attr_value)
            if (
                re.search(r"price|range", attr_name, re.IGNORECASE)
                and re.search(r"^\d+(\.\d+)?$", clean_value)
                and 1 <= int(float(clean_value)) <= 4
            ):
                price_str = "$" * int(float(clean_value))
                sentences.append(f"{attr_name}: {price_str}.")
                continue

            # Handle nested dictionaries in attributes
            if isinstance(attr_value, dict) or (isinstance(attr_value, str) and attr_value.startswith('{') and attr_value.endswith('}')):
                # Try to parse string representation of dict
                nested_dict = attr_value
                if isinstance(attr_value, str):
                    try:
                        # Replace Python-specific prefixes that might cause parsing issues
                        clean_str = attr_value.replace("u'", "'").replace('u"', '"')
                        nested_dict = eval(clean_str)  # Using eval cautiously to parse dict strings
                    except:
                        sentences.append(f"{attr_name} is {clean_text(attr_value)}.")
                        continue

                # Process the nested dictionary
                if isinstance(nested_dict, dict):
                    # Collect true values for grouping
                    true_keys = []
                    for nested_key, nested_value in nested_dict.items():
                        if nested_value == True or nested_value == "True" or nested_value == "'True'":
                            true_keys.append(nested_key.replace('_', ' '))
                        elif nested_value != False and nested_value != "False" and nested_value != "'False'" and nested_value is not None:
                            norm_value = clean_text(nested_value)
                            if norm_value:  # Skip empty values
                                sentences.append(f"{attr_name} {nested_key.replace('_', ' ')} is {norm_value}.")

                    # Add grouped true values
                    if true_keys:
                        if len(true_keys) == 1:
                            sentences.append(f"{attr_name} has {true_keys[0]}.")
                        else:
                            keys_str = ", ".join(true_keys[:-1]) + f", and {true_keys[-1]}"
                            sentences.append(f"{attr_name} has {keys_str}.")

            # Handle boolean True values
            elif attr_value == "True" or attr_value is True or attr_value == "'True'":
                sentences.append(f"Has {attr_name.replace('_', ' ')}.")

            # Handle other non-None values
            elif attr_value != "None" and attr_value is not None:
                norm_value = clean_text(attr_value)
                if norm_value:  # Skip empty values
                    sentences.append(f"{attr_name.replace('_', ' ')} is {norm_value}.")

        return sentences

    def _format_hours(self, hours_dict: Dict) -> str:
        """Format business hours in a more readable way."""
        if not hours_dict:
            return ""

        formatted_hours = []

        for day, time_range in hours_dict.items():
            # Skip closed days (0:0-0:0)
            if time_range == "0:0-0:0":
                continue

            # Format the time more nicely
            formatted_time = time_range
            if re.match(r'\d+:\d+-\d+:\d+', time_range):
                try:
                    start, end = time_range.split('-')
                    start_h, start_m = map(int, start.split(':'))
                    end_h, end_m = map(int, end.split(':'))

                    # Format with leading zeros and am/pm
                    start_ampm = "am" if start_h < 12 else "pm"
                    end_ampm = "am" if end_h < 12 else "pm"

                    # Convert to 12-hour format
                    start_h = start_h % 12
                    if start_h == 0: start_h = 12
                    end_h = end_h % 12
                    if end_h == 0: end_h = 12

                    formatted_time = f"{start_h}:{start_m:02d}{start_ampm}-{end_h}:{end_m:02d}{end_ampm}"
                except:
                    # Keep original if parsing fails
                    pass

            formatted_hours.append(f"{day}: {formatted_time}")

        if not formatted_hours:
            return ""

        return f"Hours: {', '.join(formatted_hours)}."

    def _feature_process(self, feature_name, feature_value):
        """Process a single feature into a sentence format."""
        if feature_value is None or feature_value == "":
            return ""

        # Define mapping of feature names to formatting functions
        formatters = {
            # Name and location features
            "name": lambda val: f"Name: {clean_text(val)}.",
            "city": lambda val: f"Located in {clean_text(val)}.",
            "state": lambda val: f"State: {clean_text(val)}.",
            
            # Category and rating features
            "categories": lambda val: f"Categories: {', '.join(val.split(', '))}." if val else "",
            "stars": lambda val: f"Rating: {val} stars.",
            "review_count": lambda val: f"Number of reviews: {val}.",
            
            # Complex features
            "attributes": lambda val: " ".join([sentence for sentence in self._process_attributes(val)]) if isinstance(val, dict) else "",
            "hours": lambda val: self._format_hours(val) if isinstance(val, dict) and val else "",
        }

        # Get the appropriate formatter and apply it, or apply default cleaning if no formatter exists
        formatter = formatters.get(feature_name)
        if formatter:
            return formatter(feature_value)
        else:
            # Default handling for any other feature: clean the text and add feature name
            clean_value = clean_text(feature_value)
            if clean_value:  # Skip empty values
                return f"{feature_name.replace('_', ' ').title()}: {clean_text(clean_value)}."
            return ""

    def _clean_metadata(self, business_data: Dict[str, Any]) -> str:
        """
        Process a business metadata entry into a string of descriptive sentences.
        
        Args:
            business_data: Dictionary containing business metadata
            
        Returns:
            String of sentences describing the business
        """
        meta_text = ""

        # Define important features to include
        features_needed = [
            'name', 'categories', 'stars', 'review_count',
            'city', 'state', 'attributes', 'hours'
        ]

        for feature in features_needed:
            if feature in business_data and business_data[feature] is not None:
                feature_text = self._feature_process(feature, business_data[feature])
                if feature_text:
                    meta_text += feature_text + " "

        return meta_text.strip()

    def _extract_meta_sentences(self, business_map: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract metadata sentences from a dictionary of business data.
        
        Args:
            business_map: Dictionary mapping business_id to business metadata
            
        Returns:
            Dictionary mapping business_id to metadata sentences
        """
        self.log(f"[DATASET] Extracting metadata sentences for {len(business_map)} businesses...")
        item2meta = {}

        for business_id, business_data in business_map.items():
            item2meta[business_id] = self._clean_metadata(business_data)

        return item2meta

    def _process_meta(self, business_map: Dict[str, Dict[str, Any]], output_dir: str) -> Optional[Dict[str, Any]]:
        """
        Process metadata from a business map and save to file.
        
        Args:
            business_map: Dictionary mapping business_id to business metadata
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
            item2meta = self._extract_meta_sentences(business_map)
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
        Download and process the raw Yelp dataset.
        
        This method:
        1. Downloads the Yelp reviews and business datasets
        2. Filters the data based on date range and rating score
        3. Applies K-core filtering
        4. Maps IDs to sequential integers
        5. Processes metadata
        6. Prepares the final dataset for training
        """
        # Configuration parameters
        rating_score = self.config.get('rating_score', 0.0)  # Rating score threshold for filtering
        user_core = self.config.get('user_core', 5)  # User K-core threshold
        item_core = self.config.get('item_core', 5)  # Item K-core threshold
        
        # Date range for filtering
        date_max = self.config.get('date_max', None)
        date_min = self.config.get('date_min', None)

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
        self._download_yelp_dataset(raw_data_dir)

        self.log(f'[DATASET] Loading raw data from {raw_data_dir}')
        review_data = self._load_yelp_review(raw_data_dir, date_min, date_max, rating_score)

        self.log(f'[DATASET] Yelp raw data has been processed! Reviews with rating lower than {rating_score} are filtered out')

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
        self.log(f'[DATASET] Final dataset has {len(user_count)} unique users and {len(item_count)} unique items')

        # Remap IDs
        user_items_mapped, user_num, item_num, data_maps = self._id_map(user_items)

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
        meta_data = self._load_yelp_metadata(raw_data_dir, data_maps)

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
