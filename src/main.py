from infra.amazondatasetloader import AmazonReviewsDatasetLoader
from infra.svdpredictor import SVDRatingPredictor

dataset = AmazonReviewsDatasetLoader().load_dataset()
predictor = SVDRatingPredictor(latent_dim=2)

for i in range(10):
    predictor.train(dataset)
    mse = predictor.evaluate(dataset)
    print(mse)
