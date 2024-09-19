from core.train import RatingPredictor
import numpy as np

"""
FunkSVD 알고리즘을 구현한다.
"""


class SVDRatingPredictor(RatingPredictor):
    def __init__(self, latent_dim=2):
        self.latent_dim = latent_dim
        self.dataset = None

    def train(self, dataset):        
        n_users = dataset.num_users()
        n_items = dataset.num_items()

        if self.dataset is None:
            self.dataset = dataset
            U = np.random.normal(
                scale=1.0 / self.latent_dim, size=(n_users, self.latent_dim)
            )
            self.U = U
            V = np.random.normal(
                scale=1.0 / self.latent_dim, size=(n_items, self.latent_dim)
            )
            self.V = V
        elif self.dataset != dataset:
            raise ValueError("Dataset changed")
        else:
            U = self.U
            V = self.V

        user, item, rating = dataset.to_numpy()

        epochs = 10
        gamma = 0.01  # 학습률
        lambda_ = 0.02  # 정규화 항

        for _ in range(epochs):
            for u, i, r in zip(user, item, rating):
                prediction = np.dot(U[u], V[i])
                error = r - prediction
                # 한 epoch 내에서 U[i]와 V[i]는 어차피 여러 번 업데이트된다.
                # 그러므로 여기서 U, V의 업데이트 순서를 바꿔 봐야 큰 도움이 되지 않는다.
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
