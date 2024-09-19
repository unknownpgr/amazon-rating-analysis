from core.train import RatingPredictor
import numpy as np


class SVDRatingPredictor(RatingPredictor):
    def __init__(self, latent_dim=2):
        self.latent_dim = latent_dim

    def train(self, dataset):
        n_users = dataset.num_users()
        n_items = dataset.num_items()

        # If self.U is not defined, initialize U and V
        if not hasattr(self, "U"):
            U = np.random.normal(
                scale=1.0 / self.latent_dim, size=(n_users, self.latent_dim)
            )
            self.U = U
        else:
            U = self.U

        if not hasattr(self, "V"):
            V = np.random.normal(
                scale=1.0 / self.latent_dim, size=(n_items, self.latent_dim)
            )
            self.V = V
        else:
            V = self.V

        user, item, rating = dataset.to_numpy()

        epochs = 10
        gamma = 0.01

        for epoch in range(epochs):
            for u, i, r in zip(user, item, rating):
                prediction = np.dot(U[u], V[i])
                error = r - prediction
                # 한 epoch 내에서 U[i]와 V[i]는 어차피 여러 번 업데이트된다.
                # 그러므로 여기서 U, V의 업데이트 순서를 바꿔 봐야 큰 도움이 되지 않는다.
                U[u] += gamma * error * V[i]
                V[i] += gamma * error * U[u]

    def predict(self, user_id, item_id):
        user_index = dataset.map_user_id(user_id)
        item_index = dataset.map_item_id(item_id)
        return np.dot(U[user_index], V[item_index])

    def evaluate(self, dataset):
        user, item, rating = dataset.to_numpy()
        mse = 0
        for u, i, r in zip(user, item, rating):
            prediction = np.dot(self.U[u], self.V[i])
            mse += (r - prediction) ** 2
        return mse / len(rating)
