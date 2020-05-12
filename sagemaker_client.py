import sagemaker
import boto3
import botocore
import tarfile
import os

# sageMaker session
sess = sagemaker.Session()
# iam role
role = 'arn:aws:iam::649228437072:role/service-role/AmazonSageMaker-ExecutionRole-20191205T145667'

account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/deep-q-networks'.format(account, region)

image = '649228437072.dkr.ecr.us-east-1.amazonaws.com/deep-q-networks:Latest'
bucket_name = 'sagemaker-demo-kyle'
prefix = 'deep-q-networks'
model_output = "s3://{}/{}/model".format(bucket_name, prefix)
# create an estimator
clf = sagemaker.estimator.Estimator(image,
                               role, 1, 'ml.p2.16xlarge', # ml.c4.2xlarge ml.p2.xlarge ml.p2.8xlarge ml.p2.16xlarge
                               output_path=model_output,
                               sagemaker_session=sess,
                               hyperparameters = {'DEVICE': 'cuda'})
# run entry point
clf.fit()



job_name = clf._current_job_name
# job_name = "frozen-lake-2020-05-10-02-28-09-505"
KEY = "{}/model/{}/output/model.tar.gz".format(prefix, job_name) # replace with your object key

s3 = boto3.resource('s3')

try:
    s3.Bucket(bucket_name).download_file(KEY, 'model.tar.gz')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise

tar = tarfile.open('model.tar.gz', "r:gz")
tar.extractall()
tar.close()

os.remove('model.tar.gz')