import pandas as pd
from sklearn.model_selection import train_test_split


def input_fn(target_col):

    train_dataframe = pd.read_csv('./client_repo/train.csv')
    
    x_data = train_dataframe.drop(target_col, axis=1)
    y_data = train_dataframe[target_col]
    
    x_train_client, x_test_client, y_train_client, y_test_client = train_test_split(x_data, y_data)
    
    return x_train_client, y_train_client, x_test_client, y_test_client

