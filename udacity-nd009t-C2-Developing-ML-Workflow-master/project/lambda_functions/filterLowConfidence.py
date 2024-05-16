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