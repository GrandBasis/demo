# utils/model_utils.py
import torch
import os
from nets.resnet import ResNet
from nets.vgg import VGG

def create_model(model_name, pretrained=False, device=None):
    """
    Create and return a model instance based on its name.
    """
    if model_name.startswith('resnet'):
        model = ResNet(model_name, pretrained=pretrained)
    elif model_name.startswith('vgg'):
        model = VGG(model_name, pretrained=pretrained)
    else:
        raise RuntimeError(f"Unknown model: {model_name}")
    
    if device:
        model = model.to(device)
        
    return model

def load_model_weights(model, model_name, weights_dir, device=None):
    """
    Load pretrained weights for a given model.
    """
    weights_path = os.path.join(weights_dir, f"{model_name}.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
    map_location = device if device else 'cpu'
    model.load_state_dict(torch.load(weights_path, map_location=map_location))
    return model