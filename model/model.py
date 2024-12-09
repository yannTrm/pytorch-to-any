import torch
import torch.nn as nn
from torchvision import models

from typing import Tuple, Optional

dummy_input = torch.randn(1, 3, 224, 224)

class Network(nn.Module):
    """
    Classification model using pre-trained model

    Args:
        model (str): Name of the pretrained Model we want to use (mobilenetv2, mobilenetv3-s, mobilenetv3-l, efficientnet, squeezenet, shufflenet)
        num_classes (int): Number of output classes
    """
    def __init__(self, model: str, num_classes: int):
        super(Network, self).__init__()

        #? Select the appropriate model
        if model == "mobilenetv2":
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        elif model == "mobilenetv3-s":
            self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
        
        elif model == "mobilenetv3-l":
            self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
        
        elif model == "efficientnet":
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        elif model == "squeezenet":
            self.model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
            self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        elif model == "shufflenet":
            self.model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        else:
            raise ValueError(f"Unsupported Model Name: {model}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass the model"""
        return self.model(x)
    
