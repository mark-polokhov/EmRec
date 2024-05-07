import torch.nn as nn
from torchvision.models import vgg11, VGG11_Weights


class VGGEmbedding(nn.Module):
    def __init__(self, emb_size=128):
        super().__init__()

        self.vgg_embedding = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        self.vgg_embedding.classifier[6] = nn.Linear(in_features=4096, out_features=emb_size)
    
    def forward(self, input):
        return self.vgg_embedding(input)