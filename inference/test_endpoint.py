import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sagemaker.tensorflow import TensorFlowPredictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer


def test_inference(endpoint_name, test_file_path):
    test = pd.read_csv(test_file_path)
    pre_test = data_preprocess(test)
    csv_test = pre_test.to_csv(header=False, index=False).rstrip('\n')

    predictor = TensorFlowPredictor(endpoint_name=endpoint_name,
                                    serializer=CSVSerializer(),
                                    deserializer=JSONDeserializer()
                                    )

    prediction = predictor.predict(csv_test)
    predictions_df = pd.DataFrame(prediction['predictions'], columns=['predictions'])

    result = pd.read_csv(test_file_path)
    result['Exited'] = predictions_df['predictions']

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

