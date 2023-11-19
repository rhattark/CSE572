from surprise import Dataset, Reader, SVD
import pandas as pd
from surprise.model_selection import cross_validate

file_path = 'recommender_data/ratings_small.csv'
df = pd.read_csv(file_path)
reader = Reader()
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader=reader)

algo_PMF = SVD(n_factors=100, n_epochs=20, biased=True, init_mean=0, init_std_dev=0.1, lr_all=0.005, reg_all=0.02)

cross_validate(algo_PMF, data, measures=["MAE", "RMSE"], cv=5, verbose=True)