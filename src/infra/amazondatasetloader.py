#!/usr/bin/env python
import os
from core.datasetloader import DatasetLoader, RatingDataset
from core.util import task

# Set cache directory under the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(current_dir, "cache")
os.environ["HF_HOME"] = cache_dir

from datasets import load_dataset


class AmazonReviewsDatasetLoader(DatasetLoader):
    def load_dataset(self):
        with task("Amazon review data load"):
            with task("Raw data loading"):
                # Load the dataset
                dataset = load_dataset(
                    "McAuley-Lab/Amazon-Reviews-2023",
                    "0core_timestamp_Home_and_Kitchen",
                    trust_remote_code=True,
                )["train"]

            with task("Data conversion"):
                user_ids = []
                item_ids = []
                ratings = []
                i = 0
                for row in dataset.shuffle():
                    user_ids.append(row["user_id"])
                    item_ids.append(row["parent_asin"])
                    ratings.append(float(row["rating"]))
                    i += 1
                    if i % 50000 == 0:
                        percent = i / len(dataset) * 100
                        task.log(f"Loaded {i} ratings ({percent:.2f}%)")
                    if i >= 1000000:
                        break

            return RatingDataset(user_ids, item_ids, ratings)
