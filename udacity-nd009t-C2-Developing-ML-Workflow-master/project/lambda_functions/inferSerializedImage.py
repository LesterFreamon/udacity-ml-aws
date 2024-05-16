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
