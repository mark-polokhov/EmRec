from models import ConvLayers

import torch.nn as nn
from torch.optim import Adam, SGD


class Trainer():
    def __init__(self, args):
        if args.optimizer.lower() == 'adam':
            self.optimizer = Adam
        elif args.optimizer.lower() == 'sgd':
            self.optimizer = SGD
        else:
            raise NotImplementedError
        self.criterion = nn.BCELoss()
        self.model = ConvLayers()
        self.lr_scheduler = args.lr_scheduler