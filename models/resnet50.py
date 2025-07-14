# models/resnet50.py

import torch
import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=3, pretrained=True):
    """
    Returns a ResNet50 model modified for num_classes output.
    """
    model = models.resnet50(pretrained=pretrained)
    
    # Replace final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
