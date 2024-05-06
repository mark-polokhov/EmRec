import torch.nn as nn
from torchvision.models import vgg11, VGG11_Weights


class VGGEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        self.vgg_embedding = lambda input: self.vgg.avgpool(self.vgg.features(input))
        self.emb_size = 25088
    
    def forward(self, input):
        return self.vgg_embedding(input)