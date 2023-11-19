from surprise import Dataset, Reader, SVD, KNNBasic
import pandas as pd
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np

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

def plot_similarity_comparison(user_based_mae, user_based_rmse, item_based_mae, item_based_rmse):
    similarity_metrics = ['cosine', 'msd', 'pearson']

    bar_width = 0.2
    index = np.arange(len(similarity_metrics))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # User-Based CF
    bar1 = ax.bar(index - bar_width, user_based_mae, bar_width, label='User-Based MAE')
    bar2 = ax.bar(index, user_based_rmse, bar_width, label='User-Based RMSE')
    
    # Item-Based CF
    bar3 = ax.bar(index + bar_width, item_based_mae, bar_width, label='Item-Based MAE')
    bar4 = ax.bar(index + 2*bar_width, item_based_rmse, bar_width, label='Item-Based RMSE')
    
    ax.set_xticks(index)
    ax.set_xticklabels(similarity_metrics)
    ax.set_xlabel('Similarity Metrics')
    ax.set_title('MAE and RMSE for User-Based and Item-Based Collaborative Filtering')
    ax.legend()
    
    plt.savefig('similarity_comparison_user_item.png')
    plt.show()

def collab_filter_cross_val_with_k(data, user_based, k_ip, similarity='msd'):
    sim_options = {
        'user_based': user_based,
        'name': similarity
    }
    cf = KNNBasic(k=k_ip, min_k=1, sim_options=sim_options)
    res = cross_validate(cf, data, measures=["MAE", "RMSE"], cv=5)
    return res['test_mae'].mean(), res["test_rmse"].mean()

def compare_neighbors(data, user_based):
    mae_list, rmse_list = [], []
    
    for i in range(1, 26, 1):
        mae, rmse = collab_filter_cross_val_with_k(data, user_based, i)
        mae_list.append(mae)
        rmse_list.append(rmse)

    return mae_list, rmse_list

def plot_change_in_neighbors(mae_list, rmse_list, title):
    plt.xlabel('Number of neighbors')
    plt.title(title)
    
    ax1 = plt.gca()
    ax1.set_ylabel('MAE')
    ax1.plot(range(1, 26, 1), mae_list, label='MAE', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE')
    ax2.plot(range(1, 26, 1), rmse_list, label='RMSE', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.legend()
    plt.savefig(title)
    plt.show()

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

    print('\n------------Plot similarity comparison------------\n')
    plot_similarity_comparison(user_based_mae_list, user_based_rmse_list, item_based_mae_list, item_based_rmse_list)

    # f
    print('\n------------Plot neighbor comparison - user based------------\n')
    mae_list_user_based, rmse_list_user_based = compare_neighbors(data, True)
    plot_change_in_neighbors(mae_list_user_based, rmse_list_user_based, 'Change in MAE-RMSE vs number of neighbors - User based CF')

    print('\n------------Plot neighbor comparison - item based------------\n')
    mae_list_item_based, rmse_list_item_based = compare_neighbors(data, True)
    plot_change_in_neighbors(mae_list_item_based, rmse_list_item_based, 'Change in MAE-RMSE vs number of neighbors - Item based CF')


    # g
    print('\n------------Idenify best number of neighbors------------\n')
    best_k_mae_user = np.argmin(mae_list_user_based) + 1
    best_k_rmse_user = np.argmin(rmse_list_user_based) + 1
    best_k_mae_item = np.argmin(mae_list_item_based) + 1
    best_k_rmse_item = np.argmin(rmse_list_item_based) + 1

    print(f'Best k as per MAE for User based CF: {best_k_mae_user}')
    print(f'Best k as per RMSE for User based CF: {best_k_rmse_user}')
    print(f'Best k as per MAE for Item based CF: {best_k_mae_item}')
    print(f'Best k as per RMSE for Item based CF: {best_k_rmse_item}')