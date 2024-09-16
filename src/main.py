from infra.amazondatasetloader import AmazonReviewsDatasetLoader
dataLoader = AmazonReviewsDatasetLoader()
data = dataLoader.load_data()

for i in range(10):
    print(data.get_datum(i))
    