from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

class StringIndexMap:
    def __init__(self, strings: List[str]):
        assert isinstance(strings, list)
        assert all(isinstance(string, str) for string in strings)
        unique_strings = list(set(strings))
        self.string_to_index = {string: index for index, string in enumerate(unique_strings)}
        self.index_to_string = {index: string for string, index in self.string_to_index.items()}
        self.num_strings = len(unique_strings)

    def get_index(self, string: str) -> int:
        return self.string_to_index[string]

    def get_string(self, index: int) -> str:
        return self.index_to_string[index]

    def __len__(self) -> int:
        return self.num_strings

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

      self.user_ids = user_ids
      self.item_ids = item_ids
      self.ratings = ratings

      self.user_id_map = StringIndexMap(user_ids)
      self.item_id_map = StringIndexMap(item_ids)

      self.num_users = len(self.user_id_map)
      self.num_items = len(self.item_id_map)

      self.user_indices = [self.user_id_map.get_index(user_id) for user_id in user_ids]
      self.item_indices = [self.item_id_map.get_index(item_id) for item_id in item_ids]

    def get_datum(self, index: int) -> Tuple[int, int, float]:
        return self.user_indices[index], self.item_indices[index], self.ratings[index]

class DataLoader(ABC):
    @abstractmethod
    def load_data(self) -> RatingDataset:
        pass