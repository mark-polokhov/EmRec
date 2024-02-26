from models import ConvLayers

from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from tqdm import tqdm



class Trainer():
    def __init__(self, args):
        self.model = ConvLayers()

        if args.checkpoint != None:
            self.model.load_state_dict(torch.load(args.checkpoint))

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
        loss_best = None
        for epoch in range(max_epochs):
            train_loss, train_acc = self.train_epoch(train_dataloader)
            val_loss, val_acc = self.eval(val_dataloader)
            print('Epoch {:4} | train_loss: {:4f}\ttrain_acc: {:4f}\tval_loss: {:4f}\tval_acc: {:4f}'
                  .format(epoch, train_loss, train_acc, val_loss, val_acc))
            if epoch % args.save_every == 0:
                torch.save(self.model.state_dict(), '{}_e{}_checkpoint_last.pt'
                           .format(datetime.now().strftime('%Y_%m_%d_%H_%M'), epoch))
                if loss_best == None or val_loss < loss_best:
                    torch.save(self.model.state_dict(), '{}_e{}_checkpoint_best.pt'
                               .format(datetime.now().strftime('%Y_%m_%d_%H_%M'), epoch))
                    loss_best = val_loss

    def train_epoch(self, dataloader):
        self.model.train()

        losses = 0
        length = 0

        correct = 0
        total = 0

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

            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == target).sum().item()
            total += len(target)
        
        return losses / length, correct / total
    
    def eval(self, dataloader):
        self.model.eval()

        losses = 0
        length = 0

        correct = 0
        total = 0

        for images, target in tqdm(dataloader):
            images = images.to(self.device)
            target = target.to(self.device)
            logits = self.model(images)

            loss = self.criterion(logits, target)
            losses += loss.item()
            length += 1

            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == target).sum().item()
            total += len(target)

        return losses / length, correct / total