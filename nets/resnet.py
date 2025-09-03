import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, model_name, num_classes=10, pretrained=True):
        super(ResNet, self).__init__()
        # self.base_model = models.resnet18(pretrained=True)
        self.base_model = getattr(models, model_name)(pretrained=pretrained)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
            
    def forward(self, x):
        x = self.base_model(x)
        return x