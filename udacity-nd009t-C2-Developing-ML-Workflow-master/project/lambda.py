import json
import boto3
import base64

s3 = boto3.client('s3')


def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    try:
        s3.download_file(Bucket=bucket, Key=key, Filename='/tmp/image.png')
    except Exception as e:
        print(e)
        raise e

        # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

import json
import base64
import boto3

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2024-05-14-22-09-25-785'
client = boto3.client('runtime.sagemaker')


def lambda_handler(event, context):
    # Decode the image data
    if 'body' in event and event['body'] is not None:
        event = event['body']

    encoded_image = event['image_data']
    image = base64.b64decode(encoded_image)
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType='application/x-image',
        Body=image
    )
    prediction = response['Body'].read().decode('utf-8')
    event['inferences'] = prediction

    return {
        'statusCode': 200,
        'body': {
            "image_data": event['image_data'],
            "s3_bucket": event['s3_bucket'],
            "s3_key": event['s3_key'],
            "inferences": event['inferences']

        }
    }


import json

THRESHOLD = .88


def lambda_handler(event, context):
    # Grab the inferences from the event
    if 'body' in event and event['body'] is not None:
        event = event['body']

    inferences = json.loads(event['inferences'])

    # parse inferences

    inferences = [float(i) for i in inferences]

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(inferences) > THRESHOLD

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    event['pass_thresh'] = True

    return {
        'statusCode': 200,
        'body': json.dumps(event)

    }