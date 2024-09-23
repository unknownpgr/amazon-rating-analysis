import torch
from core.train import RatingPredictor
from core.datasetloader import RatingDataset
from core.util import task
import numpy as np


class ALSRatingPredictor(RatingPredictor):
    def __init__(self, latent_dim=10, num_iterations=10, regularization=0.1):
        self.latent_dim = latent_dim
        self.num_iterations = num_iterations
        self.dataset = None
        self.user_factors = None
        self.item_factors = None

        with task("Model initialization"):
            task.log("latent_dim    :", latent_dim)
            task.log("num_iterations:", num_iterations)
            task.log("regularization:", regularization)

    def train(self, dataset: RatingDataset):
        if self.dataset is None:
            # Initialize user and item factors
            n_users = dataset.num_users()
            n_items = dataset.num_items()
            self.user_factors = torch.randn(n_users, self.latent_dim, requires_grad=True)
            self.item_factors = torch.randn(n_items, self.latent_dim, requires_grad=True)
            self.dataset = dataset
        elif self.dataset != dataset:
            raise ValueError("Dataset changed")

        user_index, item_index, rating = dataset.to_numpy()

        # Convert to PyTorch tensors
        user_index_tensor = torch.LongTensor(user_index)
        item_index_tensor = torch.LongTensor(item_index)
        rating_tensor = torch.FloatTensor(rating)

        with task("ALS Training"):
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam([self.user_factors, self.item_factors], lr=0.01)

            for iteration in range(self.num_iterations):
                user_factors = self.user_factors[user_index_tensor]
                item_factors = self.item_factors[item_index_tensor]

                # Compute the predicted ratings
                predicted_ratings = torch.matmul(user_factors, item_factors.T)

                loss = loss_fn(predicted_ratings, rating_tensor)
                loss.backward()
                optimizer.step()

                task.log(f"Iteration {iteration + 1}/{self.num_iterations}, Loss: {loss.item()}")

    def predict(self, user_id: str, item_id: str) -> float:
        user_index = self.dataset.map_user_id(user_id)
        item_index = self.dataset.map_item_id(item_id)
        return torch.dot(self.user_factors[user_index], self.item_factors[item_index]).item()

    def evaluate(self, dataset: RatingDataset) -> float:
        loss_fn = torch.nn.MSELoss()

        user_index, item_index, rating = dataset.to_numpy()
        user_index_tensor = torch.LongTensor(user_index)
        item_index_tensor = torch.LongTensor(item_index)
        rating_tensor = torch.FloatTensor(rating)

        user_factors = self.user_factors[user_index_tensor]
        item_factors = self.item_factors[item_index_tensor]
        predicted_ratings = torch.matmul(user_factors, item_factors.T)
        loss = loss_fn(predicted_ratings, rating_tensor)
        return loss.item()

    def save(self, path: str):
        torch.save({"user_factors": self.user_factors, "item_factors": self.item_factors}, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.user_factors = checkpoint["user_factors"]
        self.item_factors = checkpoint["item_factors"]

    def explain(self):
        for i in range(10):
            user_vector = self.user_factors[i].detach().numpy()
            item_vector = self.item_factors[i].detach().numpy()
            print(f"user {i}: {user_vector}")
            print(f"item {i}: {item_vector}")
            print(f"dot product: {np.dot(user_vector, item_vector)}")
            print()
