from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from core.datasetloader import RatingDataset


class RatingPredictor(ABC):
    @abstractmethod
    def train(self, dataset: RatingDataset):
        pass

    @abstractmethod
    def predict(self, user_id: str, item_id: str) -> float:
        pass

    @abstractmethod
    def evaluate(self, dataset: RatingDataset) -> float:
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def save(self, path: str):
        pass
