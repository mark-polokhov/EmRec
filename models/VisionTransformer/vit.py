import torch.nn as nn
from torchvision.models import VisionTransformer


class ViT(nn.Module):
    def __init__(self,
            img_size: int,
            num_layers: int,
            num_heads: int,
            emb_size: int,
            num_classes: int = 7):

        super(ViT, self).__init__()
        print('Training network - VisionTransformer')

        self.vision_transformer = VisionTransformer(image_size=img_size,
                                                    patch_size=16,
                                                    num_layers=num_layers,
                                                    num_heads=num_heads,
                                                    hidden_dim=emb_size,
                                                    mlp_dim=256,
                                                    dropout=0.1,
                                                    num_classes=num_classes)
    
    def forward(self, input):
        return self.vision_transformer(input)
