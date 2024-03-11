import boto3
import urllib.parse
import json
import pandas as pd
import data_pre_processing


sagemaker_runtime = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])

    segments = file_key.split('/')
    file_key_segments = segments[1].split('-')
    
    user_name = file_key_segments[0]
    task_name = file_key_segments[1]
    model_name = file_key_segments[2]
    
    file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    file_content = file_obj['Body'].read()
    preprocessed_data = data_pre_processing.data_preprocess(file_content)

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=model_name,
        Body=preprocessed_data,
        ContentType='text/csv',  
        Accept='application/json'  
    )
    result_bytes = response['Body'].read()
    result_str = result_bytes.decode('utf-8')
    result_json = json.loads(result_str)
    
    predictions = result_json['predictions']
    predictions_df = pd.DataFrame(predictions, columns=['prediction'])  
    
    input_csv_path = '/tmp/input.csv'
    result_df = pd.read_csv(input_csv_path)
    result_df['Exited'] = predictions_df['prediction']
    
    output_csv_path = '/tmp/output.csv'
    result_df.to_csv(output_csv_path, index=False)
    
    output_bucket = 'b3o-inference'
    output_key = f'output/{user_name}-{task_name}-{model_name}-output.csv'
        
    with open(output_csv_path, 'rb') as f:
        s3_client.upload_fileobj(f, output_bucket, output_key)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Inference successfully completed!')
    }

    
