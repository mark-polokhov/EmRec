import torch.nn as nn


class VGGEmbedding(nn.Module):
    def __init__(self, emb_size=128, num_classes=7):
        super().__init__()

        self.class_embedding = nn.Embedding(num_classes, emb_size)

    def forward(self, input):
        return self.class_embedding(input)