import logging
import os
import boto3
from wrapper import QcWrapper  # Import your QcWrapper class from the main code file

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def lambda_handler(event, context):
    try:
        # Get the bucket and key information from the event
        s3_event = event['Records'][0]['s3']
        bucket_name = s3_event['bucket']['name']
        object_key = s3_event['object']['key']

        # Construct the full S3 object path
        csv_file_path = f"s3://{bucket_name}/{object_key}"

        # Instantiate the QcWrapper class and call its run() method
        qc_wrapper = QcWrapper(filelist=[csv_file_path], event=event)
        success_files = qc_wrapper.run(event)

        logger.info("Successfully processed files: %s", success_files)
    except Exception as e:
        logger.error("Error occurred: %s", e)
        raise e
