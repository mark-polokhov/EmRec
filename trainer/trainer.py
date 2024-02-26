from models import ConvLayers

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from tqdm import tqdm


class Trainer():
    def __init__(self, args):
        self.model = ConvLayers()

        self.criterion = nn.CrossEntropyLoss()
        if args.optimizer.lower() == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=0.0001)
        elif args.optimizer.lower() == 'sgd':
            self.optimizer = SGD()
        else:
            raise NotImplementedError
        self.lr_scheduler = args.lr_scheduler

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('Currently used device:', self.device)

        self.model.to(self.device)

    def train(self, train_dataloader, val_dataloader, args):
        max_epochs = args.epochs
        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.eval(val_dataloader)
            print('Epoch {} train_loss: {:f}\tval_loss: {:f}'.format(epoch, train_loss, val_loss))

    def train_epoch(self, dataloader):
        self.model.train()
        losses = 0
        length = 0

        for images, target in tqdm(dataloader):
            images = images.to(self.device)
            target = target.to(self.device)
            logits = self.model(images)

            self.optimizer.zero_grad()

            loss = self.criterion(logits, target)
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
            length += 1
        
        return losses / length
    
    def eval(self, dataloader):
        self.model.eval()
        losses = 0
        length = 0

        for images, target in tqdm(dataloader):
            images = images.to(self.device)
            target = target.to(self.device)
            logits = self.model(images)

            loss = self.criterion(logits, target)
            losses += loss.item()
            length += 1

        return losses / length