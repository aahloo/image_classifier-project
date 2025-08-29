#!/usr/bin/env python3
"""
predict.py - Use a trained network to predict the class for an input image
Uses the same functions as the Jupyter notebook
"""

import argparse
import torch
import json
import numpy as np
from PIL import Image
from torchvision import models
from torch import nn
from collections import OrderedDict

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input image path')
    parser.add_argument('checkpoint', help='Checkpoint path')
    parser.add_argument('--top_k', type=int, default=1, help='Top K classes')
    parser.add_argument('--category_names', help='Category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    return parser.parse_args()

def load_checkpoint(filepath):
    # Same function as notebook
    checkpoint = torch.load(filepath, map_location='cpu')
    
    if checkpoint['architecture'] == 'vgg13':
        model = models.vgg13(weights='VGG13_Weights.DEFAULT')
    elif checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(weights='VGG16_Weights.DEFAULT')
    else:
        print(f"Architecture {checkpoint['architecture']} not recognized")
        return None
    
    for param in model.parameters():
        param.requires_grad = False
    
    input_features = checkpoint['input_features']
    hidden_layers = checkpoint['hidden_layers']
    output_size = checkpoint['output_size']
    
    classifier = nn.Sequential(OrderedDict([
        ('fc0', nn.Linear(input_features, 4096)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.4)),
        ('fc1', nn.Linear(4096, hidden_layers)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.4)),
        ('fc2', nn.Linear(hidden_layers, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    # Same function as notebook
    pil_image = Image.open(image_path)
    
    width, height = pil_image.size
    
    if width <= height:
        new_width = 256
        new_height = int(256 * (height / width))
    else:
        new_height = 256
        new_width = int(256 * (width / height))
    
    pil_image = pil_image.resize((new_width, new_height))
    
    width, height = pil_image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = left + 224
    bottom = top + 224
    
    pil_image = pil_image.crop((left, top, right, bottom))
    
    np_image = np.array(pil_image)
    np_image = np_image / 255.0
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - means) / stds
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, topk=5, gpu=False):

    # Device selection respects user's choice (GPU/CPU)
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    processed_image = process_image(image_path)
    image_tensor = torch.from_numpy(processed_image).float()
    image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    model = model.to(device)
    
    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)
        
    ps = torch.exp(output)
    top_probs, top_indices = ps.topk(topk, dim=1)
    
    top_probs = top_probs.cpu()
    top_indices = top_indices.cpu()
    
    probs = top_probs.numpy().squeeze().tolist()
    indices = top_indices.numpy().squeeze().tolist()
    
    if topk == 1:
        probs = [probs]
        indices = [indices]
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    
    return probs, classes

def main():
    args = get_input_args()

    # Device selection that respects user's --gpu flag
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Print device info for user confirmation
    if args.gpu and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif args.gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but not available. Using CPU instead.")
    else:
        print("Using CPU")
    
    model = load_checkpoint(args.checkpoint)
    if model is None:
        return
    
    probs, classes = predict(args.input, model, args.top_k)
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[cls] for cls in classes]
    else:
        class_names = classes
    
    print(f"Top {args.top_k} predictions:")
    for i, (prob, name) in enumerate(zip(probs, class_names)):
        print(f"{i+1}. {name}: {prob:.4f}")
   
if __name__ == '__main__':
    main()