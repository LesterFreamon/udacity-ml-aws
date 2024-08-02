import json
import logging
import os
import sys
import io

from PIL import ImageFile, Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

NUM_CLASSES = 133
LAYER_SIZE = 512

ImageFile.LOAD_TRUNCATED_IMAGES = True


def setup_logging() -> logging.Logger:
    """Setup logger"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logging()


def net() -> torch.nn.Module:
    """
    Create a pretrained ResNet50 model with a fully connected layer on top.
    """
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    classifier = nn.Sequential(
        nn.Linear(num_features, LAYER_SIZE),
        nn.ReLU(),
        nn.Dropout(p=0),
        nn.Linear(LAYER_SIZE, NUM_CLASSES),
    )
    model.fc = classifier
    return model


def model_fn(model_dir: str) -> torch.nn.Module:
    """Initialize a pretrained model"""
    logger.info("Loading model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = net()
    model.to(device)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))

    model.eval()
    logger.info("Finished loading model.")
    return model

def input_fn(request_body: bytes, content_type: str) -> torch.Tensor:
    logger.info(f"Received request with content type: {content_type}")
    if content_type == JPEG_CONTENT_TYPE:
        image = Image.open(io.BytesIO(request_body))
        image = image.convert('RGB')  # Ensure image is in RGB format
        return image
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def output_fn(prediction_output, content_type: str):
    logger.info(f'Generating response with content type: {content_type}')
    if content_type == JSON_CONTENT_TYPE:
        response = prediction_output.cpu().numpy().tolist()  # Convert to list
        return json.dumps(response)  # Serialize as JSON
    else:
        raise ValueError(f"Unsupported content type for output: {content_type}")

def predict_fn(input_object, model):
    logger.info("Preparing input for prediction.")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Match training resize
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Match training normalization
        ]
    )

    logger.info("Transforming input.")
    input_tensor = transform(input_object)
    
    with torch.no_grad():
        logger.info("Generating prediction.")
        prediction = model(input_tensor.unsqueeze(0))  # Model expects batch dimension

    return prediction