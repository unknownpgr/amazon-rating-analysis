#!/usr/bin/env python
import os
from core.dataloader import DataLoader, RatingDataset

# Set cache directory under the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(current_dir, "cache")
os.environ["HF_HOME"] = cache_dir

from datasets import load_dataset

class AmazonReviewsDatasetLoader(DataLoader):
    def load_data(self):

        # Load the dataset
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "0core_timestamp_Home_and_Kitchen",
            trust_remote_code=True,
        )['train']

        user_ids = []
        item_ids = []
        ratings = []
        
        i=0
        for row in dataset:
            user_ids.append(row["user_id"])
            item_ids.append(row["parent_asin"])
            ratings.append(float(row["rating"]))
            i+=1
            if i%100000==0:
                percent = i/len(dataset)*100
                print(f"Loaded {i} ratings ({percent:.2f}%)")

        return RatingDataset(user_ids, item_ids, ratings) 