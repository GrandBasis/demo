import argparse
import torch
import torch.nn as nn
import random
import os
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image

from utils.model_utils import create_model, load_model_weights
from utils.data_utils import get_cifar10_transforms, CIFAR10_MEAN, CIFAR10_STD
from utils.common_utils import get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for distinguishable sample generation.")
    parser.add_argument('--model', type=str, required=True, help='Pretrained model for generating distinguishable samples.')
    parser.add_argument('--exp_tag', type=str, required=True, help='Unique experiment identifier used to create results directory.')
    return parser.parse_args()

def epsilon_norm(x_input, mean, std):
    std_tensor = torch.tensor(std).to(x_input.device).view(1, -1, 1, 1) 
    epsilon_bound =  0.2 / std_tensor
    return epsilon_bound

def bound_norm(x_input, mean, std):
    mean_tensor = torch.tensor(mean).to(x_input.device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std).to(x_input.device).view(1, -1, 1, 1) 
    lower_bound = (torch.zeros_like(x_input) - mean_tensor) / std_tensor
    upper_bound = (torch.ones_like(x_input) - mean_tensor) / std_tensor
    return lower_bound, upper_bound

def compute_prob(logits, t):

    softmax_output = F.softmax(logits, dim=1)
    # Get batch size
    batch_size = logits.size(0)
    # Extract target class scores Z_t
    Z_t = softmax_output[torch.arange(batch_size), t].view(-1, 1)
    # Create mask excluding target class
    mask = torch.ones_like(softmax_output, dtype=torch.bool)
    mask[torch.arange(batch_size), t] = False
    # Set target class scores to -inf for max exclusion
    masked_softmax = softmax_output.masked_fill(~mask, float('-inf'))
    # Compute max non-target class score max_Z_i
    max_Z_i, _ = masked_softmax.max(dim=1, keepdim=True)

    prob_confidence = Z_t - max_Z_i

    return prob_confidence.item()


def compute_f(x, t, model, tau):
    # Ensure model in evaluation mode
    model.eval()
    
    # Forward pass to obtain logits
    logits = model(x)
    softmax_output = F.softmax(logits, dim=1)

    # Get batch size
    batch_size = x.size(0)
    
    # Extract target class scores Z_t
    Z_t = softmax_output[torch.arange(batch_size), t].view(-1, 1)

    # Create mask excluding target class
    mask = torch.ones_like(softmax_output, dtype=torch.bool)
    mask[torch.arange(batch_size), t] = False
    
    # Set target class scores to -inf for max exclusion
    masked_softmax = softmax_output.masked_fill(~mask, float('-inf'))
    
    # Compute max non-target class score max_Z_i
    max_Z_i, _ = masked_softmax.max(dim=1, keepdim=True)

    # Compute f(x, t) = clamp(Z_t - max_Z_i, min=tau)
    f_x_t = torch.max(Z_t - max_Z_i, tau)
    # f_x_t = torch.max(Z_t, tau)
    return f_x_t

def update_x(x, t, model, tau, alpha, iteration, mean, std):
    # Create detached copy of input
    x_dis = x.detach()
    for iter in range(iteration):
        x_dis.requires_grad = True
        f_x_t = compute_f(x_dis, t, model, tau)
        # Maximize confidence margin
        loss = f_x_t.mean()
        # Backpropagate gradients
        loss.backward()
        # x_dis = x_dis - alpha * torch.sign(x_dis.grad)
        x_dis = x_dis - alpha * torch.sign(x_dis.grad)
        # x_dis = x_dis + alpha * x_dis.grad
        epsilon_bound = epsilon_norm(x_dis, mean, std)
        lower_bound, upper_bound = bound_norm(x_dis, mean, std)
        x_dis = torch.max(torch.min(x_dis, x + epsilon_bound), x - epsilon_bound)
        x_dis = torch.max(torch.min(x_dis, upper_bound), lower_bound)
        x_dis = x_dis.detach() 
        adv_output = model(x_dis)
        # print(compute_prob(adv_output, t))
        if compute_prob(adv_output, t) < tau:
            break
    return x_dis

def denormalize(tensor, mean, std):
    result = torch.zeros_like(tensor)
    for i in range(tensor.size(0)):
        result[i] = tensor[i] * std[i] + mean[i]
    return result


if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    
    MEAN = CIFAR10_MEAN
    STD = CIFAR10_STD
    transform = get_cifar10_transforms(train=False)        
    testset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    
    # Randomly select a subset of the test set
    random_indices = random.sample(range(len(testset)), 2000)
    selected_data = [(testset[i][0], testset[i][1]) for i in random_indices]
    
    # Create the model and load its weights
    model = create_model(args.model, pretrained=False, device=device)
    model = load_model_weights(model, args.model, 'models', device=device)
    model.eval()

    exp_tag = os.path.join(args.exp_tag, "samples", f'{args.model}')
    os.makedirs(exp_tag, exist_ok=True)
    
    correct = 0
    total = 0
    tau = 0.5
    alpha = 0.0002
    iteration = 1000
    sample_info = []  # Stores sample metadata (original/target/predicted labels)

    tau = torch.tensor(tau).to(device)

    # Generate adversarial samples
    for idx, (data, original_label) in enumerate(selected_data):
        data = data.unsqueeze(0).to(device)
        original_label = torch.tensor([original_label], device=device)

        output_adv = model(data)
        _, output_pred = torch.max(output_adv, 1)

        if output_pred.item() == original_label.item():

            dis_data = update_x(data, original_label, model, tau, alpha, iteration, MEAN, STD)
            
            output = model(dis_data)
            _, predicted = torch.max(output, 1)

            # Check if prediction remains unchanged (non-transferable sample)
            if predicted.item() == original_label.cpu().item():
                tau_prime = 0.5
                prob_diff = compute_prob(output, original_label.cpu().item())

                print(f"prob_diff:{prob_diff},tau_prime:{tau_prime}")
                if predicted == original_label and prob_diff < tau_prime and prob_diff > tau_prime - 0.2:

                    # Convert to original pixel space
                    dis_data_denorm = denormalize(dis_data.squeeze(0).cpu(), mean=MEAN, std=STD)
                    # Save as PNG with metadata in filename
                    filename = f"sample_{idx}_orig_{original_label.item()}_target_{original_label.item()}_pred_{predicted.item()}.png"
                    file_path = os.path.join(exp_tag, filename)
                    save_image(dis_data_denorm, file_path)
                    correct += 1
                    
                    if correct == 100:
                        break
