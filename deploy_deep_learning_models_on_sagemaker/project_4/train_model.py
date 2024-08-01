#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

from .config import NUM_CLASSES, HOME_DIR

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass

def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # I want to do transfer learning on resnet152. I will freeze the weights of the base model. Then I will add a new fully connected layer with NUM_CLASSES outputs
    model = models.resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    last_layer = resnet152.fc
    num_features = resnet152.fc.in_features

    # Add a new, untrained layer that will be trained
    classifier = nn.Sequential(nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(256, NUM_CLASSES))

    model.fc = classifier
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    pass

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()

    train_loader, test_loader = create_data_loaders(args.data, args.batch_size)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--fc_layer_size', type=int, default=128)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    
    args=parser.parse_args()
    
    main(args)