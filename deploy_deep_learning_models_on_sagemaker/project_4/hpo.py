# TODO: Import your dependencies.
# For instance, below are some dependencies you might need if you are using Pytorch
import os
import sys
import logging
import time
import numpy as np
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile

import argparse

NUM_CLASSES = 133

ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
logger = setup_logging()



def test(model, test_loader, criterion, device):
    """
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    """
    hook = get_hook(create_if_not_exists=False)
    logger.info("Start testing")

    if hook:
        hook.set_mode(modes.EVAL)
        hook.register_loss(criterion)

    
    hook.register_forward_hook("all")  # Capture all forward passes during evaluation


    model.eval()
    model.to(device)

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    average_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    hook.save_scalar("accuracy", accuracy)
    hook.save_scalar("average_loss", average_loss)

    return average_loss, accuracy




def train(model, train_loader, valid_loader, criterion, optimizer, args, device):
    """
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    """
    hook = get_hook(create_if_not_exists=True)

    epoch_times = []

    if hook:
        hook.register_loss(criterion)

    logger.info("Start training")

    for i in range(args.epochs):
        start = time.time()
        logger.info(f"Epoch {i + 1}/{args.epochs}")
        print(f"Epoch {i + 1}/{args.epochs}")
        if hook:
            hook.set_mode(modes.TRAIN)

        train_loss = 0
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        logger.info(f"Train Loss: {train_loss / len(train_loader)}")
        logger.info("Start Validating")
        if hook:
            hook.set_mode(modes.EVAL)

        model.eval()
        val_loss = 0

        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += torch.eq(predicted, labels).sum().item()
            logger.info(f"Validation Loss: {val_loss / len(valid_loader)}")
            logger.info(f"Validation Accuracy: {correct / total}")

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        logger.info(
            f"Epoch {i + 1}/{args.epochs}: train loss {train_loss / len(train_loader)}, val loss {val_loss / len(valid_loader)}, in {epoch_time} sec"
        )
        print(
            f"Epoch {i + 1}/{args.epochs}: train loss {train_loss / len(train_loader)}, val loss {val_loss / len(valid_loader)}, in {epoch_time} sec"
        )
    return model
    
    


def net(layer_size: int):
    """Create the base model used for transfer learning and attach additional layers"""

    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_features = (
        model.fc.in_features
    )  # Get the number of input features of the last layer of the base model
    classifier = nn.Sequential(
        nn.Linear(num_features, layer_size),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(layer_size, NUM_CLASSES),
    )
    model.fc = classifier
    return model


def create_data_loaders(data_dir, batch_size, is_train):
    """
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transform,
            ]
        )

    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def main(args):
    """
    TODO: Initialize a model by calling the net function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")
    model = net(layer_size=args.fc_layer_size)
    model.to(device)  # Move model to the correct device

    train_dir = os.getenv('SM_CHANNEL_TRAIN')
    valid_dir = os.getenv('SM_CHANNEL_VALIDATION')
    test_dir = os.getenv('SM_CHANNEL_TEST')

    logger.info(f"Creating data loaders")

    train_loader = create_data_loaders(train_dir, args.batch_size, is_train=True)
    logger.info(f"Train loader has {len(train_loader)} batches")

    val_loader = create_data_loaders(valid_dir, args.batch_size, is_train=False)
    logger.info(f"Validation loader has {len(val_loader)} batches")

    test_loader = create_data_loaders(test_dir, args.batch_size, is_train=False)
    logger.info(f"Test loader has {len(test_loader)} batches")

    """
    TODO: Create your loss and optimizer
    """
    loss_criterion = nn.CrossEntropyLoss()
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    """
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    """
    model = train(model, train_loader, val_loader, loss_criterion, optimizer, args, device)

    """
    TODO: Test the model to see its accuracy
    """
    test_loss, test_accuracy = test(model, test_loader, loss_criterion, device)

    logger.info(f"Test Loss: {test_loss}")
    logger.info(f"Test Accuracy: {test_accuracy}")

    """
    TODO: Save the trained model
    """
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    TODO: Specify all the hyperparameters you need to use to train your model.
    """
    parser.add_argument('--s3_data_path', type=str, default='s3://udacity-ml-aws/dogImages',
                        help='S3 path to the training data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--fc_layer_size', type=int, default=128)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))


    args = parser.parse_args()

    main(args)
