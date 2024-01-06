import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ConvLayers(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.layers.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=7),
            nn.Sigmoid(),
        )
    
    def forward(self, input):
        return self.layers(input)