# utils/common_utils.py
import torch
import os
import json

def get_device():
    """
    Get the available computing device (CUDA or CPU).
    Returns:
        torch.device: The selected device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_json_result(result_dict, file_path):
    """
    Write a result dictionary to a JSON file, appending it as a new line.
    """
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(file_path, 'a+', encoding='utf-8') as file_obj:        
        file_obj.write(str(json.dumps(result_dict, ensure_ascii=False)) + '\n')