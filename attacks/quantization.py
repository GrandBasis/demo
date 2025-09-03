import argparse
import torch
import os

def quantize_model(model, num_bits=8):
    scale = 2 ** num_bits - 1  # Quantization range
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:  # Only quantize weight matrices
                param_min = param.min()
                param_max = param.max()
                # Normalize weights [0, scale]
                param -= param_min  
                param /= (param_max - param_min) 
                param *= scale
                param.round_()  # Round to nearest integer
                # Denormalize to original range
                param /= scale
                param *= (param_max - param_min)
                param += param_min
    return model

         