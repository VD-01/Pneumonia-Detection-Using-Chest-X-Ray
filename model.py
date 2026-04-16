import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def get_resnet_model(num_classes=2, pretrained=True):
    """
    Returns a ResNet18 model modified for the specified number of classes.
    """
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
        
    num_ftrs = model.fc.in_features
    # We replace the last fully connected layer
    # Since we have 2 classes: NORMAL and PNEUMONIA
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    model = get_resnet_model(num_classes=2)
    print(model)
