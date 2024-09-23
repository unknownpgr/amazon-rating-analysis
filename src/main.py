from infra.amazondatasetloader import AmazonReviewsDatasetLoader
from infra.svdpredictor import SVDRatingPredictor
from infra.alspredictor import ALSRatingPredictor
from core.util import task

with task("Amazon Review Data Analysis"):

    with task("Dataset Loading"):
        dataset = AmazonReviewsDatasetLoader().load_dataset()
        for i in range(10):
            task.log(dataset.get_datum(i))
        train, test = dataset.split([0.8, 0.2])

    predictor = ALSRatingPredictor(latent_dim=2)

    with task("Training"):
        for i in range(10):
            with task(f"Epoch {i + 1}"):
                predictor.train(train)
                train_mse = predictor.evaluate(train)
                test_mse = predictor.evaluate(test)
                task.log(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
