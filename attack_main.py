import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import os

from attacks.fine_tune import fine_tune_model
from attacks.quantization import quantize_model
from utils.model_utils import create_model, load_model_weights
from utils.data_utils import get_cifar10_transforms
from utils.common_utils import get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for model attack.")
    parser.add_argument('--model', type=str, required=True, choices=['resnet50','vgg19'], help='Target pretrained model for attack evaluation.')
    parser.add_argument('--exp_tag', type=str, required=True, help='Unique experiment identifier used to create results directory.')
    parser.add_argument('--attack', type=str, required=True, choices=['fine_tune','quantization'], help='Attack methodology to execute.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    
    # Create the model and load its original pre-trained weights
    model = create_model(args.model, pretrained=False, device=device)
    model = load_model_weights(model, args.model, 'models', device=device)
    
    # Apply the specified attack method
    if args.attack == "fine_tune":
        # Fine-tune the model on a subset of the training data
        transform = get_cifar10_transforms(train=True)   
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        subset_trainset = Subset(trainset, list(range(5000)))
        trainloader = DataLoader(subset_trainset, batch_size=128, shuffle=True)     
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)    
        # Apply fine_tune to the model            
        model = fine_tune_model(model, trainloader, criterion, optimizer, device, epochs=10)  
    elif args.attack == "quantization":
        # Apply quantization to the model
        model = quantize_model(model, num_bits=8)
    else:
        raise RuntimeError("Attack type not found")  
    
    # Save the state dictionary of the attacked model
    exp_tag = os.path.join(args.exp_tag, "attacks", args.attack)
    os.makedirs(exp_tag, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(exp_tag, f"{args.model}.pth"))