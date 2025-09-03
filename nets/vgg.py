import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, model_name, num_classes=10, pretrained=True):
        super(VGG, self).__init__()
        # self.base_model = models.vgg19(pretrained=True)
        self.base_model = getattr(models, model_name)(pretrained=pretrained)
        self.base_model.classifier[6] = nn.Linear(self.base_model.classifier[6].in_features, num_classes)
        
    def forward(self, x):
        x = self.base_model(x)
        return x