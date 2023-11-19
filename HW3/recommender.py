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

def compare_similarity_metrics(data, user_based):
    mae_list = []
    rmse_list = []
    titles = ['Cosine similarity:\n', '\nMean Squared Difference similarity:\n', '\nPearson similarity:\n']
    similarity_metrics = ['cosine', 'msd', 'pearson']

    for title, similarity in zip(titles, similarity_metrics):
        print(title)
        test_mae, test_rmse = collab_filter_cross_val(data, user_based, similarity)
        mae_list.append(test_mae)
        rmse_list.append(test_rmse)
        print(f'\nMAE Test: {test_mae} \nRMSE Test: {test_rmse}')

    return mae_list, rmse_list

if __name__ == '__main__':
    # a
    file_path = 'recommender_data/ratings_small.csv'
    data = load_file(file_path)

    # b is info only

    # c
    print('\n------------PMF------------\n')
    test_mae, test_rmse = pmf(data)
    print(f'\nMAE Test: {test_mae} \nRMSE Test: {test_rmse}')

    print('\n------------User based CF------------\n')
    test_mae, test_rmse = collab_filter_cross_val(data, True)
    print(f'\nMAE Test: {test_mae} \nRMSE Test: {test_rmse}')

    print('\n------------Item based CF------------\n')
    test_mae, test_rmse = collab_filter_cross_val(data, False)
    print(f'\nMAE Test: {test_mae} \nRMSE Test: {test_rmse}')

    # d is in pdf

    # e
    print('\n------------User based CF Similarity Metrics Comparison------------\n')
    user_based_mae_list, user_based_rmse_list = compare_similarity_metrics(data, True)

    print('\n------------Item based CF Similarity Metrics Comparison------------\n')
    item_based_mae_list, item_based_rmse_list = compare_similarity_metrics(data, False)
