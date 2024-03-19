import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def input_fn(file_path, target_col):
    train_dataframe = pd.read_csv(file_path)

    x_data = train_dataframe.drop(target_col, axis=1)
    y_data = train_dataframe[target_col]
    pre_x_train = data_preprocess(x_data)

    x_train_client, x_test_client, y_train_client, y_test_client = train_test_split(pre_x_train, y_data)

    return x_train_client, y_train_client, x_test_client, y_test_client


def data_preprocess(dataframe):
    label_encoder = LabelEncoder()
    for column in dataframe.columns:
        if dataframe[column].dtype == 'object':
            dataframe[column] = label_encoder.fit_transform(dataframe[column])

    scaler = StandardScaler()
    numerical_data = dataframe.select_dtypes(include=['int64', 'float64'])
    scaled_numerical_data = scaler.fit_transform(numerical_data)
    dataframe[numerical_data.columns] = scaled_numerical_data

    return dataframe