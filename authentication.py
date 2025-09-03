import argparse
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from utils.model_utils import create_model
from utils.data_utils import get_cifar10_transforms
from utils.common_utils import get_device, write_json_result

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for authentication.")
    parser.add_argument('--surrogate_model', type=str, required=True, choices=['resnet50','vgg19'], help='Surrogate model used for attack generation.')
    parser.add_argument('--target_model', type=str, required=True, choices=['resnet50','vgg19'], help='Target model to evaluate attack against.')
    parser.add_argument('--attack', type=str, required=True, choices=['fine_tune','quantization'], help='Attack method name.')
    parser.add_argument('--exp_tag', type=str, required=True, help='Unique experiment identifier used to create results directory.')
    return parser.parse_args()

class CustomCIFARDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # List all PNG files in the directory
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        # Parse the filename to extract labels
        parts = img_name[:-4].split('_')  # Remove '.png' extension and split
        sample_id = int(parts[1])
        orig_label = int(parts[3])
        target_label = int(parts[5])
        # You can also extract pred_label if needed: pred_label = int(parts[7])
        
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, orig_label, target_label

if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    module_name = os.path.splitext(os.path.basename(__file__))[0]
        
    transform = get_cifar10_transforms(train=False)
    
    # Load the custom dataset of generated samples
    data_path = os.path.join(args.exp_tag, "samples", f'{args.surrogate_model}')
    dataset_obj = CustomCIFARDataset(root_dir=data_path, transform=transform)
    dataloader = DataLoader(dataset_obj, batch_size=1, shuffle=False, num_workers=2)                   
                    
    # Create the target model instance
    target_model = create_model(args.target_model, pretrained=False, device=device)
    
    # Load the weights of the *attacked* target model
    target_model_path = os.path.join(args.exp_tag, "attacks", args.attack, f"{args.target_model}.pth")
    target_model.load_state_dict(torch.load(target_model_path, map_location=device))
    target_model.eval()

    result_dict = {
        "surrogate_model": args.surrogate_model,
        "target_model": args.target_model,
    }   

    correct = 0
    total = 0
        
    # Evaluate the target model's performance on the generated samples
    with torch.no_grad():
        for images, orig_labels, target_labels in dataloader:
            images = images.to(device)
            target_labels = target_labels.to(device)
            
            outputs = target_model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
            
    # Calculate the accuracy (or Attack Success Rate)
    accuracy = 100 * correct / total 
    result_dict[args.attack] = round(accuracy, 4)
        
    # Write the final result to a JSON file
    write_json_result(result_dict, os.path.join(args.exp_tag, f"{module_name}.json"))