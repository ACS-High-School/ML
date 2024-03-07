import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sagemaker.tensorflow import TensorFlowPredictor


def test_inference(endpoint_name, test_file_path):
    test = pd.read_csv(test_file_path)
    pre_test = data_preprocess(test)

    predictor = TensorFlowPredictor(endpoint_name)
    prediction = predictor.predict(pre_test)

    result = pd.read_csv(test_file_path)
    result['Exited'] = prediction

    display(result)


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

