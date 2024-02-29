import sagemaker
from sagemaker import get_execution_role, image_uris
from sagemaker.model import Model
from datetime import datetime


def retrieve_container_image(region, framework, version, instance_type):
    container = image_uris.retrieve(region=region,
                                    framework=framework,
                                    version=version,
                                    image_scope='inference',
                                    instance_type=instance_type)
    return container


def create_model(container, s3_bucket, model_s3_key, role):
    model_url = f's3://{s3_bucket}/{model_s3_key}'
    model = Model(image_uri=container,
                  model_data=model_url,
                  role=role)
    return model


def create_endpoint_name():
    endpoint_name = f"DEMO-{datetime.utcnow():%Y-%m-%d-%H%M}"
    print("EndpointName =", endpoint_name)
    return endpoint_name


def deploy_model(model, instance_type, initial_instance_count, endpoint_name):
    predictor = model.deploy(
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    return predictor
