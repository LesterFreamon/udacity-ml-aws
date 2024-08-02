# TODO: Import your dependencies.
# For instance, below are some dependencies you might need if you are using Pytorch
import os
import sys
import logging
import time
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile

import argparse

NUM_CLASSES = 133

ImageFile.LOAD_TRUNCATED_IMAGES = True


def setup_logging() -> logging.Logger:
    """Setup Logger"""
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


def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Tests the given model using the provided test loader and criterion on the specified device.
    Returns both the average test loss and the accuracy percentage.
    Parameters:
        model (torch.nn.Module): The PyTorch model to be tested.
        test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        criterion (torch.nn.modules.loss._Loss): Loss function to evaluate the model.
        device (torch.device): The device (CPU or GPU) on which to perform the testing.

    Returns:
        average_loss (float): The average loss computed over all test batches.
        accuracy (float): The percentage of correctly predicted instances over the test set.
    """
    hook = get_hook(create_if_not_exists=False)
    logger.info("Start testing")

    if hook:
        hook.set_mode(modes.EVAL)
        hook.register_loss(criterion)

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
    if hook:
        hook.save_scalar("accuracy", accuracy)
        hook.save_scalar("average_loss", average_loss)

    logger.info(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)"
    )

    return average_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    """
    Trains the given model using the provided training and validation loaders, criterion, and optimizer on the specified device.
    """
    hook = get_hook(create_if_not_exists=True)

    epoch_times = []

    if hook:
        hook.register_loss(criterion)

    logger.info("Start training")

    for i in range(args.epochs):
        start = time.time()
        logger.info(f"Epoch {i + 1}/{args.epochs}")
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
    return model


def net(args):
    """Create the base model used for transfer learning and attach additional layers"""

    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_features = (
        model.fc.in_features
    )  # Get the number of input features of the last layer of the base model
    classifier = nn.Sequential(
        nn.Linear(num_features, args.fc_layer_size),
        nn.ReLU(),
        nn.Dropout(p=args.dropout_rate),
        nn.Linear(args.fc_layer_size, NUM_CLASSES),
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
    model = net(args)
    model.to(device)  # Move model to the correct device

    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.register_hook(model)

    train_dir = os.getenv("SM_CHANNEL_TRAIN")
    valid_dir = os.getenv("SM_CHANNEL_VALIDATION")
    test_dir = os.getenv("SM_CHANNEL_TEST")

    logger.info(f"Creating data loaders")

    train_loader = create_data_loaders(train_dir, args.batch_size, is_train=True)
    logger.info(f"Train loader has {len(train_loader)} batches")

    val_loader = create_data_loaders(valid_dir, args.batch_size, is_train=False)
    logger.info(f"Validation loader has {len(val_loader)} batches")

    test_loader = create_data_loaders(test_dir, args.batch_size, is_train=False)
    logger.info(f"Test loader has {len(test_loader)} batches")

    # Create your loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    if args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # train the model
    model = train(
        model, train_loader, val_loader, loss_criterion, optimizer, args, device
    )

    # Test the model to see its accuracy
    test_loss, test_accuracy = test(model, test_loader, loss_criterion, device)

    logger.info(f"Test Loss: {test_loss}")
    logger.info(f"Test Accuracy: {test_accuracy}")

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    TODO: Specify all the hyperparameters you need to use to train your model.
    """
    parser.add_argument(
        "--s3_data_path",
        type=str,
        default="s3://udacity-ml-aws/dogImages",
        help="S3 path to data",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--fc_layer_size", type=int, default=128)
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )

    args = parser.parse_args()

    main(args)
