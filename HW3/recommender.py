from surprise import Dataset, Reader, SVD, KNNBasic
import pandas as pd
from surprise.model_selection import cross_validate

def load_file(file_path):
    print('Reading file...')
    df = pd.read_csv(file_path)
    reader = Reader()
    print('Loading file...')
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader=reader)
    return data

def pmf(data):
    algo_PMF = SVD(n_factors=100, n_epochs=20, biased=True, init_mean=0, init_std_dev=0.1, lr_all=0.005, reg_all=0.02)
    res = cross_validate(algo_PMF, data, measures=["MAE", "RMSE"], cv=5)
    return res['test_mae'].mean(), res["test_rmse"].mean()

def collab_filter_cross_val(data, user_based, similarity='msd'):
    sim_options = {
        'user_based': user_based,
        'name': similarity
    }
    cf = KNNBasic(k=40, min_k=1, sim_options=sim_options)
    res = cross_validate(cf, data, measures=["MAE", "RMSE"], cv=5)
    return res['test_mae'].mean(), res["test_rmse"].mean()

if __name__ == '__main__':
    file_path = 'recommender_data/ratings_small.csv'
    data = load_file(file_path)

    print('\n------------PMF------------\n')
    test_mae, test_rmse = pmf(data)
    print(f'\nMAE Test: {test_mae} \nRMSE Test: {test_rmse}')

    print('\n------------User based CF------------\n')
    test_mae, test_rmse = collab_filter_cross_val(data, True)
    print(f'\nMAE Test: {test_mae} \nRMSE Test: {test_rmse}')

    print('\n------------Item based CF------------\n')
    test_mae, test_rmse = collab_filter_cross_val(data, False)
    print(f'\nMAE Test: {test_mae} \nRMSE Test: {test_rmse}')

    print('\n------------User based CF Similarity Metrics Comparison------------\n')
    print('Cosine similarity:\n')
    test_mae, test_rmse = collab_filter_cross_val(data, True, 'cosine')
    print(f'\nMAE Test: {test_mae} \nRMSE Test: {test_rmse}')

    print('\nMean Squared Difference similarity:\n')
    test_mae, test_rmse = collab_filter_cross_val(data, True, 'msd')
    print(f'\nMAE Test: {test_mae} \nRMSE Test: {test_rmse}')

    print('\nPearson similarity:\n')
    test_mae, test_rmse = collab_filter_cross_val(data, True, 'pearson')
    print(f'\nMAE Test: {test_mae} \nRMSE Test: {test_rmse}')