import torch
import torch.nn as nn
# from torchvision.models import VisionTransformer
# from vit_pytorch import ViT as ViTPyTorch
from pytorch_pretrained_vit import ViT as PretrainedViT


class ViT(nn.Module):
    def __init__(self,
            img_size: int,
            num_layers: int,
            num_heads: int,
            emb_size: int,
            num_classes: int = 7):

        super(ViT, self).__init__()
        print('Training network - VisionTransformer')

        # self.vision_transformer = VisionTransformer(image_size=img_size,
        #                                             patch_size=16,
        #                                             num_layers=num_layers,
        #                                             num_heads=num_heads,
        #                                             hidden_dim=emb_size,
        #                                             mlp_dim=256,
        #                                             dropout=0.1,
        #                                             num_classes=num_classes)
            
        # vit_pytorch
        # self.vision_transformer = ViTPyTorch(
        #     image_size = 128,
        #     patch_size = 32,
        #     num_classes = 7,
        #     dim = 1024,
        #     depth = 6,
        #     heads = 8,
        #     mlp_dim = 2048
        # )

        # DINO
        # self.vision_transformer = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        # self.vision_transformer.load_state_dict(torch.load('./models/VisionTransformer/dino_deitsmall16_pretrain_full_checkpoint.pth'))

        # DINO v2
        # self.vision_transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # self.classifier = nn.Sequential(
        #     nn.Linear(384, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 7)
        # )

        # Pretrained
        self.vision_transformer = PretrainedViT('B_16_imagenet1k',
                                                pretrained=True,
                                                image_size=img_size,
                                                num_classes=num_classes)
    
    def forward(self, input):
        return self.vision_transformer(input)
