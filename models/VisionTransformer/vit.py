import torch
import torch.nn as nn
from torchvision.models import VisionTransformer
from vit_pytorch import ViT as ViTPyTorch
from pytorch_pretrained_vit import ViT as PretrainedViT

# from torchvision.models import resnet50, ResNet50_Weights

# from vit_pytorch.distill import DistillableViT, DistillWrapper


from transformers import ViTForImageClassification


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
        #     image_size = img_size,
        #     patch_size = 16,
        #     num_classes = 7,
        #     dim = 768,
        #     depth = 3,
        #     heads = 4,
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

        # Distillated
        # self.teacher = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # self.teacher.fc = nn.Sequential(
        #     nn.Linear(in_features=2048, out_features=7),
        #     nn.Sigmoid(),
        # )
        # self.vision_transformer = DistillableViT(
        #     image_size = img_size,
        #     patch_size = 32,
        #     num_classes = num_classes,
        #     dim = 1024,
        #     depth = num_layers,
        #     heads = num_heads,
        #     mlp_dim = 2048,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )
        # self.distiller = DistillWrapper(
        #     student = self.vision_transformer,
        #     teacher = self.teacher,
        #     temperature = 3,           # temperature of distillation
        #     alpha = 0.5,               # trade between main loss and distillation loss
        #     hard = False               # whether to use soft or hard distillation
        # )
    
        # Huggingface
        # model_name_or_path = 'google/vit-base-patch16-224-in21k'
        # self.vision_transformer = ViTForImageClassification.from_pretrained(
        #     model_name_or_path,
        #     num_labels=num_classes)

    def forward(self, input, labels=None):
        return self.vision_transformer(input)
