import pandas as pd
import numpy as np
import io


def data_preprocess(file_content):
    dataframe = pd.read_csv(io.BytesIO(file_content))
    
    input_csv_path = '/tmp/input.csv'  
    dataframe.to_csv(input_csv_path, index=False)

    geography_mapping = {label: idx for idx, label in enumerate(dataframe['Geography'].unique())}
    gender_mapping = {label: idx for idx, label in enumerate(dataframe['Gender'].unique())}
    surname_mapping = {label: idx for idx, label in enumerate(dataframe['Surname'].unique())}

    dataframe['Geography'] = dataframe['Geography'].map(geography_mapping)
    dataframe['Gender'] = dataframe['Gender'].map(gender_mapping)
    dataframe['Surname'] = dataframe['Surname'].map(surname_mapping)

    num_features = ['CustomerId', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    numeric_data = dataframe[num_features].values

    scaler = lambda x: (x - np.mean(x)) / np.std(x)
    standardized_data = np.apply_along_axis(scaler, 0, numeric_data)

    dataframe[num_features] = standardized_data
    preprocessed_csv = dataframe.to_csv(index=False, header=False)
    
    return preprocessed_csv
