# ML

- 다수의 클라이언트는 연합학습을 통해 모델 성능 개선 및 SOTA 모델 배포
- 서비스 사용자들은 배포된 모델 기반으로 codeless Inference 수행

<br>
<br>

## Features
### 연합학습 Workflow 구축 및 서비스 구현
- Step Functions, SQS, SNS, Lambda 를 활용하여 연합학습 Workflow 구축
- Sagemaker 를 활용하여 Training Pipeline 구현
- S3, DynamoDB 를 활용하여 Model 정보 및 Task Token 저장소 구축
- 연합학습을 위한 Python Package 생성 및 배포

<br>
  
### Inference Workflow 구축 및 서비스 구현
- Sagemaker 를 활용하여 real-time inference 를 위한 model endpoint 배포
- Lambda를 활용하여 real-time inference invoke endpoint 구현
- S3 를 활용하여 Client input data, output data 저장소 구축

<br>
<br>
 
## Technologies
- [Python](https://www.python.org/downloads/release/python-31013/) 3.10.13
- [Tensorflow](https://www.tensorflow.org/versions/r2.12/api_docs/python/tf?_gl=1*1m7gk30*_up*MQ..*_ga*NTAxMzMwMjkzLjE3MTM3NTQ0MjE.*_ga_W0YLR4190T*MTcxMzc1NDQyMS4xLjAuMTcxMzc1NDQ1My4wLjAuMA..) 2.12.1
- [AWS Sagemaker-distribution](https://github.com/aws/sagemaker-distribution/blob/main/build_artifacts/v1/v1.6/v1.6.0/RELEASE.md) 1.6.0
- AWS Lambda
- AWS S3
- AWS Step Functions
- AWS SQS
- AWS SNS
- AWS DynamoDB

<br>
<br>

## Prerequisites

### AWS CDK
[AWS CDK Docs](https://docs.aws.amazon.com/cdk/)

- AWS CloudFormation template 생성
```bash
cdk synth
```
- CDK app 배포
```bash
cdk deploy
```
- 배포된 환경과 로컬 CDK app 차이
```bash
cdk diff
```
- AWS CloudFormation stack delete
```bash
cdk destroy
```

<br>
<br>

## Trouble Shooting 
### 사용자 친화적이지 않은 연합학습 실행 로직
- 문제 원인
  - 연합학습을 시작하기 위해 코드에 대한 이해 필요
  - 환경 셋팅, 데이터 셋 생성, 학습 모델 정의, 연합 학습 등의 대량의 코드 파일 존재

- 해결 방안
  - 연합학습을 위한 Python Package 생성 및 배포
  - Package에서 필요한 모듈을 import 하여 연합학습 진행
  - Package name : b3o-fedlearn
  - Package 링크: https://pypi.org/project/b3o-fedlearn/

<br>

### 연합학습을 위한 Jupyter Lab space 생성 및 실행 시, 학습 환경 수동 구축
- 문제 원인
  - space instance 생성 시, Sagemaker Distribution을 image로 사용
  - Amazon Sagemaker Distribution은 미리 정의된 Python 패키지 버전을 포함하여 배포

- 해결 방안
  - JupyterLab Lifecycle Configurations를 사용하여 학습 환경 자동 구축
  - Lifecycle Configuration 스크립트 작성을 하여 필요한 특정 버전 패키지 설치
  - Lifecycle Configuration 스크립트 작성을 하여 Git repo clone
  - install package : numpy==1.23.1, b3o-fedlearn==1.0.0
  - Git repo : https://github.com/ACS-High-School/ML_Client.git

<br>
<br>

## Feature improvements
### Inference 를 위한 Endpoint 4개 존재
- 개선 이유
  - Model 에 따른 Endpoint 4개 생성시, 비용이 4배
  
- 개선 방안
  - Multi-Container Endpoint 생성 후 Inference 로직 구현하여 비용 절감
  
<br>

### Client를 위한 연합학습 Jupyterlab 정적할당
- 개선 이유
  - Client 수 및 Jupyterlab space를 미리 정의하여 진행하므로, 확장성을 고려하지 못함

- 개선 방안
  - Lambda를 통해 Jupyterlab space 생성 및 삭제 API 구현
  - RDS, Lambda를 통해 Client 정보를 전달받은 후, Step Functions 에 할당하여 실행

<br>
<br>


## Contribution
- 🫠 [정다영](https://github.com/Dayoung-Jung)
