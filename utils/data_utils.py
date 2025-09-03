# utils/data_utils.py
from torchvision import transforms

# Mean and standard deviation for CIFAR-10 dataset normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def get_cifar10_transforms(train=True):
    """
    Get the data transformations for the CIFAR-10 dataset.
    """
    
    if train:
        # For training, include data augmentation
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        # For testing, only perform normalization
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])