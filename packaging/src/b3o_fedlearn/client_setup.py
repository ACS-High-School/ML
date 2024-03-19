import os

def setup():

    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('client_repo'):
        os.makedirs('client_repo')

    model_py_content = """
import tensorflow as tf

class MLMODEL:
    def __init__(self):
        self.model = tf.keras.models.Sequential() # write your model code
        self.model.compile() # write your model compile code

    def getModel(self):
        return self.model
"""

    with open('client_repo/MODEL.py', 'w') as f:
        f.write(model_py_content)

    client_json_content = """
{
    "s3_fl_model_registry": "flserverdatabucketcdk0602",
    "dynamodb_table_model_info": "fltaskscdk0602", 
    "clients": 
    [
        {
            "member_ID": "1",
            "sqs_region": "us-east-1",
            "client_queue_name": "clientsqs-us-east-1"
        },
        {
            "member_ID": "2",
            "sqs_region": "us-east-2",
            "client_queue_name": "clientsqs-us-east-2"
        },
        {
            "member_ID": "3",
            "sqs_region": "us-west-1",
            "client_queue_name": "clientsqs-us-west-1"
        },
        {
            "member_ID": "4",
            "sqs_region": "us-west-2",
            "client_queue_name": "clientsqs-us-west-2"
        }
    ]
}
"""
    with open('client_repo/client_config.json', 'w') as f:
        f.write(client_json_content)
