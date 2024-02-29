from tensorflow import keras
import tensorflow as tf
import numpy as np
import tarfile
import shutil
import boto3
import os

from code_repo import MODEL


def download_weights_from_s3(bucket, file_key, local_file_path):
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, file_key, local_file_path)


def setup_and_save_model(local_weights_file, model_save_path):
    mlmodel = MODEL.MLMODEL()
    model = mlmodel.getModel()
    weights = np.load(local_weights_file, allow_pickle=True)
    model.set_weights(weights)
    tf.keras.models.save_model(model, model_save_path, save_format='tf')
    os.remove(local_weights_file)


def compress_and_upload_model_to_s3(source_dir, target_dir, bucket, object_key):
    os.makedirs(target_dir, exist_ok=True)

    for file_name in os.listdir(source_dir):
        shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

    with tarfile.open(f'{object_key}', 'w:gz') as tar:
        tar.add(target_dir, arcname=os.path.basename(target_dir))

    s3_client = boto3.client('s3')
    s3_client.upload_file(f'{object_key}', bucket, object_key)

    os.remove(object_key)
