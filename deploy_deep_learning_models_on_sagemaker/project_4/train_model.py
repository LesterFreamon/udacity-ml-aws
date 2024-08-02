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


def test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    hook: Any,
) -> Tuple[float, float]:
    """
    Test the model on a the test dataset and regurn the test loss and accuracy.
    """
    logger.info("Start testing")

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
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    hook: Any,
) -> nn.Module:
    """
    Train the model on the training dataset and validate on the validation dataset.
    """

    hook.register_loss(criterion)

    for epoch in range(args.epochs):
        start = time.time()
        logger.info(f"Epoch: {epoch+1}/{args.epochs}")
        hook.set_mode(modes.TRAIN)
        model.train()
        train_loss = 0.0
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

        hook.set_mode(modes.EVAL)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            logger.info(f"Val Loss: {val_loss / len(valid_loader)}")
            logger.info(f"Val Accuracy: {correct / len(valid_loader.dataset)}")

        epoch_time = time.time() - start
        logger.info(f"Epoch Time: {epoch_time}")
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs}: train loss {train_loss / len(train_loader)}, val loss {val_loss / len(valid_loader)}, in {epoch_time} sec"
        )

    logger.info("Training complete")

    return model


def net(args):
    """
    Create a pretrained ResNet50 model with a fully connected layer on top.
    """
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


def create_data_loader(data_dir: str, batch_size: int, is_train: bool) -> DataLoader:
    """
    Create a data loader for the given data directory and batch size.
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def create_data_loaders(
    args: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create a train, validation and test data loaders"""
    train_dir = os.getenv("SM_CHANNEL_TRAIN")
    valid_dir = os.getenv("SM_CHANNEL_VALIDATION")
    test_dir = os.getenv("SM_CHANNEL_TEST")

    logger.info(f"Creating data loaders")

    train_loader = create_data_loader(train_dir, args.batch_size, is_train=True)
    logger.info(f"Train loader has {len(train_loader)} batches")

    val_loader = create_data_loader(valid_dir, args.batch_size, is_train=False)
    logger.info(f"Validation loader has {len(val_loader)} batches")

    test_loader = create_data_loader(test_dir, args.batch_size, is_train=False)
    logger.info(f"Test loader has {len(test_loader)} batches")
    return train_loader, val_loader, test_loader


def optimizer_fn(args: Dict[str, Any], model: nn.Module) -> optim.Optimizer:
    """Crete an optimizer for the given model"""
    if args.optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def main(args):
    """
    The main function that runs the training loop and saves the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")
    model = net(args)
    model.to(device)

    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.register_hook(model)

    # Create your loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(args, model)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(args)

    # train model
    model = train(
        model, train_loader, val_loader, loss_criterion, optimizer, args, device, hook
    )

    # Test the model to see its accuracy
    test_loss, test_accuracy = test(model, test_loader, loss_criterion, device, hook)
    logger.info(f"Test Loss: {test_loss}")
    logger.info(f"Test Accuracy: {test_accuracy}")

    # Save the trained model
    logger.info(f"Saving the model to {args.model_dir}")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Specify any training args
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
