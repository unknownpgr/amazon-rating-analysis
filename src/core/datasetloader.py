from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
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


class DatasetLoader(ABC):
    @abstractmethod
    def load_dataset(self) -> RatingDataset:
        pass
