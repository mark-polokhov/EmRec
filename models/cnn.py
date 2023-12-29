import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ConvLayers(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, classes=7)