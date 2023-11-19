from surprise import Dataset, Reader, SVD, KNNBasic
import pandas as pd
from surprise.model_selection import cross_validate

file_path = 'recommender_data/ratings_small.csv'
print('Reading file...')
df = pd.read_csv(file_path)
reader = Reader()
print('Loading file...')
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader=reader)

print('\n------------PMF------------\n')
algo_PMF = SVD(n_factors=100, n_epochs=20, biased=True, init_mean=0, init_std_dev=0.1, lr_all=0.005, reg_all=0.02)
cross_validate(algo_PMF, data, measures=["MAE", "RMSE"], cv=5, verbose=True)

print('\n------------User based CF------------\n')
user_based_CF_sim_options = {
    'user_based': True
}
algo_user_based_CF = KNNBasic(k=40, min_k=1, sim_options=user_based_CF_sim_options)
cross_validate(algo_user_based_CF, data, measures=["MAE", "RMSE"], cv=5, verbose=True)

print('\n------------Item based CF------------\n')
item_based_CF_sim_options = {
    'user_based': False
}
algo_item_based_CF = KNNBasic(k=40, min_k=1, sim_options=item_based_CF_sim_options)
cross_validate(algo_item_based_CF, data, measures=["MAE", "RMSE"], cv=5, verbose=True)