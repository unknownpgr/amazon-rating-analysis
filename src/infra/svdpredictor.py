from core.train import RatingPredictor
from core.util import task
import numpy as np

"""
FunkSVD 알고리즘을 구현한다.
"""


class SVDRatingPredictor(RatingPredictor):
    def __init__(self, latent_dim=2):
        self.latent_dim = latent_dim
        self.dataset = None

        self.gamma = 0.05  # 학습률
        self.lambda_ = 0.02

        with task("Model initialization"):
            task.log("latent_dim :", latent_dim)
            task.log("gamma      :", self.gamma)

    def train(self, dataset):
        n_users = dataset.num_users()
        n_items = dataset.num_items()

        if self.dataset is None:
            self.dataset = dataset
            self.U = np.random.normal(scale=1.0 / self.latent_dim, size=(n_users, self.latent_dim))
            self.V = np.random.normal(scale=1.0 / self.latent_dim, size=(n_items, self.latent_dim))
        elif self.dataset != dataset:
            raise ValueError("Dataset changed")

        U = self.U
        V = self.V

        user, item, rating = dataset.to_numpy()

        epochs = 10
        gamma = self.gamma
        lambda_ = self.lambda_

        for _ in range(epochs):
            for u, i, r in zip(user, item, rating):
                prediction = np.dot(U[u], V[i])
                error = r - prediction
                U[u] += gamma * (error * V[i] - lambda_ * U[u])
                V[i] += gamma * (error * U[u] - lambda_ * V[i])

    def predict(self, user_id, item_id):
        user_index = self.dataset.map_user_id(user_id)
        item_index = self.dataset.map_item_id(item_id)
        return np.dot(self.U[user_index], self.V[item_index])

    def evaluate(self, dataset):
        user, item, rating = dataset.to_numpy()
        mse = 0
        for u, i, r in zip(user, item, rating):
            prediction = np.dot(self.U[u], self.V[i])
            mse += (r - prediction) ** 2
        return mse / len(rating)

    def save(self, path):
        np.savez(path, U=self.U, V=self.V)

    def load(self, path):
        data = np.load(path)
        self.U = data["U"]
        self.V = data["V"]

    def explain(self):
        for i in range(10):
            user_vector = self.U[i]
            item_vector = self.V[i]
            print(f"user {i}: {user_vector}")
            print(f"item {i}: {item_vector}")
            print(f"dot product: {np.dot(user_vector, item_vector)}")
            print()
