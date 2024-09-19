from infra.amazondatasetloader import AmazonReviewsDatasetLoader
from infra.svdpredictor import SVDRatingPredictor

dataset = AmazonReviewsDatasetLoader().load_dataset()

for i in range(100):
    print(dataset.get_datum(i))

train, test = dataset.split([0.8, 0.2])
predictor = SVDRatingPredictor(latent_dim=2)

for i in range(10):
    predictor.train(train)
    train_mse = predictor.evaluate(train)
    test_mse = predictor.evaluate(test)
    print(f"Epoch {i + 1}, Train MSE: {train_mse}, Test MSE: {test_mse}")