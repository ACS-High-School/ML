import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def input_fn(member_ID, file_path, test_size_per_client):
    train_dataframe = pd.read_csv(file_path)
    pre_train = data_preprocess(train_dataframe)
    x_data = pre_train.drop('Exited', axis=1)
    y_data = pre_train['Exited']
    x_train_client, x_test_client, y_train_client, y_test_client = train_test_split(x_data,
                                                                                    y_data,
                                                                                    test_size=test_size_per_client)

    return x_train_client, y_train_client, x_test_client, y_test_client


def data_preprocess(dataframe):
    label_encoder = LabelEncoder()
    dataframe['Geography'] = label_encoder.fit_transform(dataframe['Geography'])
    dataframe['Gender'] = label_encoder.fit_transform(dataframe['Gender'])
    dataframe['Surname'] = label_encoder.fit_transform(dataframe['Surname'])

    num_features = ['CustomerId', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                    'IsActiveMember', 'EstimatedSalary']
    scaler = StandardScaler()
    dataframe[num_features] = scaler.fit_transform(dataframe[num_features])

    return dataframe