from .vgg_embedding import VGGEmbedding
from .positional_embedding import PositionalEncoding

import torch.nn as nn
from torch.nn import Transformer


class VGGTransformer(nn.Module):
    def __init__(self,
            num_encoder_layers: int,
            num_decoder_layers: int,
            emb_size: int,
            nhead: int,
            num_classes: int = 7,
            dim_feedforward: int = 512,
            dropout: float = 0.1):

        super(VGGTransformer, self).__init__()
        print('Training network - VGGTransformer')

        self.embedding = VGGEmbedding(emb_size=emb_size)
        self.positional_embedding = PositionalEncoding(emb_size, dropout)
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.classifier = nn.Linear(emb_size, num_classes)
    
    def forward(self, input, target):
        print(input.shape, target.shape)
        input_positional_embedding = self.positional_embedding(self.embedding(input))
        print(input_positional_embedding.shape)
        output = self.transformer(input_positional_embedding, target)
        return self.classifier(output)

