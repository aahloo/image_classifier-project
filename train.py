#!/usr/bin/env python3
"""
train.py - Train a new network on a dataset and save the model as a checkpoint
Uses the same functions as the Jupyter notebook
"""

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Path to dataset')
    parser.add_argument('--save_dir', default='.', help='Directory to save checkpoint')
    parser.add_argument('--arch', default='vgg13', choices=['vgg13', 'vgg16'], help='Architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    return parser.parse_args()

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Same transforms as notebook
    training_dt = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    validation_dt = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    testing_dt = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    training_ds = datasets.ImageFolder(train_dir, transform=training_dt)
    validation_ds = datasets.ImageFolder(valid_dir, transform=validation_dt)
    testing_ds = datasets.ImageFolder(test_dir, transform=testing_dt)
    
    dataLoaders = {
        'training': torch.utils.data.DataLoader(training_ds, batch_size=64, shuffle=True),
        'validation': torch.utils.data.DataLoader(validation_ds, batch_size=64, shuffle=True),
        'testing': torch.utils.data.DataLoader(testing_ds, batch_size=64, shuffle=True)
    }
    
    return dataLoaders, training_ds.class_to_idx

def model_setup(hidden, class_to_idx, lr, arch='vgg13'):
    # Same function as notebook
    if arch == 'vgg13':
        model = models.vgg13(weights='VGG13_Weights.DEFAULT')
    elif arch == 'vgg16':
        model = models.vgg16(weights='VGG16_Weights.DEFAULT')
    
    for param in model.parameters():
        param.requires_grad = False
    
    input_features = model.classifier[0].in_features
    output_size = 102
    
    classifier = nn.Sequential(OrderedDict([
        ('fc0', nn.Linear(input_features, 4096)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.4)),
        ('fc1', nn.Linear(4096, hidden)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.4)),
        ('fc2', nn.Linear(hidden, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    return model, criterion, optimizer

def model_validate(model, criterion, dataLoader):
    # Same function as notebook
    test_loss = 0
    accuracy = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in dataLoader:
            if torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                inputs, labels = inputs.to('cpu'), labels.to('cpu')
            
            output = model.forward(inputs)
            test_loss += criterion(output, labels).item()
            
            ps = torch.exp(output)
            top_prob, top_class = ps.topk(1, dim=1)
            equals = top_class == (labels.view(*top_class.shape))
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    valid_loss, valid_acc = test_loss / len(dataLoader), accuracy / len(dataLoader)
    return valid_loss, valid_acc

def model_train(model, num_epochs, criterion, optimizer, dataloaders_training, dataloaders_validation):
    # Same function as notebook
    print("Data training process now underway...")
    
    steps = 0
    print_every = 10
    device = 'cpu'
    
    if torch.cuda.is_available():
        device = 'cuda'
    
    model.to(device)
    
    for e in range(num_epochs):
        running_loss = 0
        
        for images, labels in iter(dataloaders_training):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            training_loss = running_loss / print_every
            steps += 1
            
            if steps % print_every == 0:
                test_loss, accuracy = model_validate(model, criterion, dataloaders_validation)
                
                print("Epoch: {} / {} ".format(e+1, num_epochs),
                      "Training Loss: {:.4f} ".format(training_loss),
                      "Validation Loss: {:.4f} ".format(test_loss),
                      "Accuracy: {:.4f} ".format(accuracy))
                
                running_loss = 0
                model.train()

def main():
    args = get_input_args()
    
    dataLoaders, class_to_idx = load_data(args.data_dir)
    model, criterion, optimizer = model_setup(args.hidden_units, class_to_idx, args.learning_rate, args.arch)
    
    model_train(model, args.epochs, criterion, optimizer, dataLoaders['training'], dataLoaders['validation'])
    
    # Save checkpoint with same format as notebook
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'architecture': args.arch,
        'hidden_layers': args.hidden_units,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'optimizer_state': optimizer.state_dict(),
        'input_features': model.classifier[0].in_features,
        'output_size': 102
    }
    
    checkpoint_path = args.save_dir + '/checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    main()