import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import numpy as np
import json
import warnings
from .client_fedlearn import FLClient
import time
import os
import glob
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def FL_start(member_ID, config_client, x_train_client, y_train_client, x_test_client, y_test_client):

    start_time = 0
    end_time = 0
    hasStarted = True
    global_model = None
    
    with open(config_client, 'r') as config_file:
        client_config = json.load(config_file)
    s3_fl_model_registry = client_config["s3_fl_model_registry"]
    dynamodb_table_model_info = client_config["dynamodb_table_model_info"]
    member_ID = int(client_config["clients"][member_ID-1]["member_ID"])
    sqs_region = client_config["clients"][member_ID-1]["sqs_region"]
    client_queue_name = client_config["clients"][member_ID-1]["client_queue_name"]

    cross_account = False
    if (
        "cross_account_sqs_role" in client_config.keys() and 
        "cross_account_s3_role" in client_config.keys() and
        "cross_account_dynamodb_role" in client_config.keys() 
       ):
        cross_account = True
    else:
        assert("cross_account_sqs_role" not in client_config.keys() and 
               "cross_account_s3_role" not in client_config.keys() and
               "cross_account_dynamodb_role" not in client_config.keys())
    
    client = FLClient(member_ID, x_train_client, y_train_client, x_test_client, y_test_client)
    
    signalTerminate = False
    while not signalTerminate:
        if cross_account == True:
            messages = client.receiveNotificationsFromServer(sqs_region, client_queue_name, client_config["cross_account_sqs_role"])
        else:
            messages = client.receiveNotificationsFromServer(sqs_region, client_queue_name)
    
        transactions = []
        curr_round = -1
        for msg in messages:
            msg_body = msg.body
            msg_body_json = json.loads(msg_body)
            msg_rec = msg_body_json["Message"]
            transaction_dict = json.loads(msg_rec)  
    
            if type(transaction_dict) == str:
                transaction_dict = json.loads(transaction_dict)
                if transaction_dict['roundId'] == "NA":
                    if cross_account == True:
                        global_model = client.downloadFLGlobalModel(s3_fl_model_registry, client_config["cross_account_s3_role"])
                    else: 
                        global_model = client.downloadFLGlobalModel(s3_fl_model_registry)
                    end_time = time.monotonic()
                    print("FL training is finished")
                    signalTerminate = True
                    break             
            
            transaction = transaction_dict["Input"]
            transaction["TaskToken"] = transaction_dict["TaskToken"]
            transactions.append(transaction)
            
            if int(transaction['roundId']) > curr_round:
                curr_round = transaction['roundId']
            
        if len(transactions) > 0:
            s3_client = boto3.client('s3')
            file_path = 'client_repo/MODEL.py'
            bucket_name = 'b3o'
            s3_file_path = f'fedlearn/model/{transaction["Task_Name"]}-model.py'

            try:
                s3_client.head_object(Bucket=bucket_name, Key=s3_file_path)
            except ClientError as e:
                error_code = int(e.response['Error']['Code'])
                if error_code == 404:
                    s3_client.upload_file(file_path, bucket_name, s3_file_path)
                else:
                    print(f"Error occurred: {e}")


            infoServer = client.processGlobalModelInfoFromServer(transactions)

            if (infoServer != None):
                if hasStarted:
                    start_time = time.monotonic()
                    hasStarted = False
                print("FL training round: " + str(curr_round) + "\n")
                print("1: Receive notification")            
                printDiction = {
                        "taskName": transaction["Task_Name"],
                        "taskId": transaction["Task_ID"],
                        "roundId":  transaction["roundId"],
                        "memberId": transaction["member_ID"],
                        "numClientEpochs": transaction["numClientEpochs"],
                        "trainAcc": float(transaction["trainAcc"]) if transaction["trainAcc"] != "NA" else "NA",
                        "testAcc" : float(transaction["testAcc"]) if transaction["testAcc"] != "NA" else "NA", 
                        "trainLoss": float(transaction["trainLoss"]) if transaction["trainLoss"] != "NA" else "NA", 
                        "testLoss" : float(transaction["testLoss"]) if transaction["testLoss"] != "NA" else "NA", 
                        "weightsFile": transaction["weightsFile"],
                        "numClientsRequired": transaction["numClientsRequired"],
                        "source": transaction["source"],
                        "taskToken": transaction["TaskToken"],
                    }                        
                print("Received notification: {} \n".format(str(printDiction))) 
                
                global_model_name = infoServer["weightsFile"]
            
                if cross_account == True:
                    local_model_name, local_model_info  = client.localTraining(global_model_name, s3_fl_model_registry, client_config["cross_account_s3_role"])
                else:
                    local_model_name, local_model_info = client.localTraining(global_model_name, s3_fl_model_registry)
                
                taskToken = infoServer["TaskToken"]
                local_model_info["TaskToken"] = taskToken
    
                print("4: Upload local model & its information ...")
                if cross_account == True:
                    client.uploadToFLServer(s3_fl_model_registry, 
                                            local_model_name, 
                                            dynamodb_table_model_info, 
                                            local_model_info, 
                                            client_config["cross_account_s3_role"], 
                                            client_config["cross_account_dynamodb_role"]
                                           )
                else:
                    client.uploadToFLServer(s3_fl_model_registry, 
                                            local_model_name, 
                                            dynamodb_table_model_info,
                                            local_model_info)
    
                client.roundId = client.roundId + 1

    

  

