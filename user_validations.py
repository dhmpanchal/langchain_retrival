import boto3
import os

try:
    aws_access_key_id=os.getenv('AWS_S3_ACCESS_KEY')
    print("accesss key" ,aws_access_key_id)
except Exception:
    aws_access_key_id=''
try:
    aws_secret_access_key=os.getenv('AWS_S3_SECRET_KEY')
    print("aws_secret_access_key key" ,aws_secret_access_key)
except Exception:
    aws_secret_access_key=''





    