from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from core.util import task
import numpy as np


class StringIndexMap:
    def __init__(self, strings: List[str]):
        assert isinstance(strings, list)
        assert all(isinstance(string, str) for string in strings)
        unique_strings = list(set(strings))
        self.__init(unique_strings)

    def __init(self, unique_strings: List[str]):
        self.__string_to_index = {
            string: index for index, string in enumerate(unique_strings)
        }
        self.__index_to_string = unique_strings
        self.__num_strings = len(unique_strings)

    def get_index(self, string: str) -> int:
        return self.__string_to_index[string]

    def get_string(self, index: int) -> str:
        return self.__index_to_string[index]

    def __len__(self) -> int:
        return self.__num_strings


class RatingDataset:
    def __init__(self, user_ids: List[str], item_ids: List[str], ratings: List[float]):
        with task("Dataset initialization"):
            # Type check assertions
            assert isinstance(user_ids, list)
            assert isinstance(item_ids, list)
            assert isinstance(ratings, list)
            assert all(isinstance(user_id, str) for user_id in user_ids)
            assert all(isinstance(item_id, str) for item_id in item_ids)
            assert all(isinstance(rating, float) for rating in ratings)
            assert len(user_ids) == len(item_ids) == len(ratings)

            # Mappings
            self.__user_id_map = StringIndexMap(user_ids)
            self.__item_id_map = StringIndexMap(item_ids)

            # Core data
            self.__user_indices = np.array(
                [self.__user_id_map.get_index(user_id) for user_id in user_ids],
                dtype=np.int32,
            )
            self.__item_indices = np.array(
                [self.__item_id_map.get_index(item_id) for item_id in item_ids],
                dtype=np.int32,
            )
            self.__ratings = np.array(ratings, dtype=np.float32)

    def get_datum(self, index: int) -> Tuple[int, int, float]:
        user_index = self.__user_indices[index]
        item_index = self.__item_indices[index]
        rating = self.__ratings[index]
        user_id = self.__user_id_map.get_string(user_index)
        item_id = self.__item_id_map.get_string(item_index)
        return user_id, item_id, rating

    def map_user_id(self, user_id: str) -> int:
        return self.__user_id_map.get_index(user_id)

    def map_item_id(self, item_id: str) -> int:
        return self.__item_id_map.get_index(item_id)

    def restore_user_id(self, user_index: int) -> str:
        return self.__user_id_map.get_string(user_index)

    def restore_item_id(self, item_index: int) -> str:
        return self.__item_id_map.get_string(item_index)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.__user_indices, self.__item_indices, self.__ratings

    def num_users(self) -> int:
        return len(self.__user_id_map)

    def num_items(self) -> int:
        return len(self.__item_id_map)

    def __len__(self) -> int:
        return len(self.__ratings)

    def split(self, ratios: List[float]) -> List["RatingDataset"]:
        assert isinstance(ratios, list)
        assert all(isinstance(ratio, float) for ratio in ratios)

        data_counts = []
        for ratio in ratios:
            data_counts.append(int(len(self) * ratio))

        while sum(data_counts) < len(self):
            data_counts[0] += 1

        datasets = []
        start_index = 0
        for data_count in data_counts:
            end_index = start_index + data_count
            sub_user_indices = self.__user_indices[start_index:end_index]
            sub_item_indices = self.__item_indices[start_index:end_index]
            sub_ratings = self.__ratings[start_index:end_index]
            sub_dataset = RatingDataset([], [], [])
            sub_dataset.__user_indices = sub_user_indices
            sub_dataset.__item_indices = sub_item_indices
            sub_dataset.__ratings = sub_ratings
            sub_dataset.__user_id_map = self.__user_id_map
            sub_dataset.__item_id_map = self.__item_id_map
            datasets.append(sub_dataset)
            start_index = end_index

        return datasets


class DatasetLoader(ABC):
    @abstractmethod
    def load_dataset(self) -> RatingDataset:
        pass
