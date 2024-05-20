from models import ConvLayers, VGGTransformer, ViT

from collections.abc import Iterable
from datetime import datetime
import os
import re
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from tqdm import tqdm

import wandb


class Trainer():
    def __init__(self, args):
        self.model_name = args.model.lower()
        if self.model_name == 'resnet50':
            self.model = ConvLayers()
        elif self.model_name == 'vggtransformer':
            assert args.num_encoder_layers > 0, 'num_encoder_layers must be > 0'
            assert args.num_decoder_layers > 0, 'num_decoder_layers must be > 0'
            assert args.num_heads > 0, 'num_heads must be > 0'
            self.model = VGGTransformer(num_encoder_layers=args.num_encoder_layers,
                                        num_decoder_layers=args.num_decoder_layers,
                                        emb_size=args.emb_size, nhead=args.num_heads)
        elif self.model_name in {'vit', 'visiontransformer'}:
            self.model = ViT(img_size=args.img_size,
                            num_layers=args.num_encoder_layers,
                            num_heads=args.num_heads,
                            emb_size=args.emb_size)
        else:
            raise NotImplementedError

        self.loaded_epoch = 0
        if args.checkpoint:
            self.model.load_state_dict(torch.load(''.join(['./checkpoint/', args.checkpoint])))
            self.loaded_epoch = int(re.findall(r'_e\d+_', args.checkpoint)[0][2:-1])

        self.criterion = nn.CrossEntropyLoss() # needs to be chosen carefully
        if args.optimizer.lower() == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=1e-5)
        elif args.optimizer.lower() == 'sgd':
            self.optimizer = SGD()
        else:
            raise NotImplementedError
        self.lr_scheduler = args.lr_scheduler

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('Currently using device:', self.device)

        self.model.to(self.device)
        self.save_config(args)

        self.acc_best = None

        self.config_descr = f'{args.transform}_img{args.img_size}_b{args.batch_size}'

        wandb.init(
            project="EmRec",
            config=args.__dict__
        )

    def train(self, train_dataloader, val_dataloader, args):
        max_epochs = args.epochs
        for epoch in range(max_epochs):
            train_loss, train_acc = self.train_epoch(train_dataloader)
            val_loss, val_acc = self.eval(val_dataloader)
            print('Epoch {:4} [{:4}] | train_loss: {:4f}\ttrain_acc: {:4f}\tval_loss: {:4f}\tval_acc: {:4f}'
                  .format(epoch + 1, self.loaded_epoch + epoch + 1, train_loss, train_acc, val_loss, val_acc))

            wandb.log({'train_accuracy': train_acc, 'train_loss': train_loss, 'val_accuracy': val_acc, 'val_loss': val_loss,
                       'epoch': self.loaded_epoch + epoch + 1})

            if epoch % args.save_every == 0:
                self.save_checkpoint(epoch, val_acc)

    def train_epoch(self, dataloader):
        self.model.train()

        losses = 0
        length = 0

        correct = 0
        total = 0

        for images, target in tqdm(dataloader):
            images = images.to(self.device)
            target = target.to(self.device)
            if self.model_name == 'resnet50':
                logits = self.model(images)
            elif self.model_name == 'vggtransformer':
                logits = self.model(images, target)
            elif self.model_name == 'vit' or self.model_name == 'visiontransformer':
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
            wandb.log({'running_accuracy': (predicted == target).sum().item() / len(target), 'running_loss': loss.item()})
        
        return losses / length, correct / total
    
    def eval(self, dataloader):
        self.model.eval()

        losses = 0
        length = 0

        correct = 0
        total = 0

        with torch.no_grad():
            for images, target in tqdm(dataloader):
                images = images.to(self.device)
                target = target.to(self.device)
                if self.model_name == 'resnet50':
                    logits = self.model(images)
                elif self.model_name == 'vggtransformer':
                    logits = self.model(images, target)
                elif self.model_name == 'vit' or self.model_name == 'visiontransformer':
                    logits = self.model(images)

                loss = self.criterion(logits, target)
                losses += loss.item()
                length += 1

                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == target).sum().item()
                total += len(target)

        return losses / length, correct / total
    
    def save_checkpoint(self, epoch, val_acc):
        '''
        Saves checkpoint of the model
        '''
        epoch += self.loaded_epoch + 1
        checkpoint_last = [checkpoint for checkpoint in os.listdir('./checkpoint/')
                           if checkpoint.endswith('checkpoint_last.pt')]
        try:
            torch.save(self.model.state_dict(), ''.join(['./checkpoint/', '{}_{}_e{}_checkpoint_last.pt'
                            .format(datetime.now().strftime('%Y%m%d'), self.config_descr, epoch)]))
        except OSError:
            print('checkpoint_last failed to save')
            return
        [os.remove(''.join(['./checkpoint/', old_checkpoint])) for old_checkpoint in checkpoint_last]
        if self.acc_best == None or val_acc > self.acc_best:
            checkpoint_best = [checkpoint for checkpoint in os.listdir('./checkpoint/')
                               if checkpoint.endswith('checkpoint_best.pt')]
            try:
                torch.save(self.model.state_dict(), ''.join(['./checkpoint/', '{}_{}_e{}_checkpoint_best.pt'
                            .format(datetime.now().strftime('%Y%m%d'), self.config_descr, epoch)]))
            except OSError:
                print('checkpoint_best failed to save')
                return
            self.acc_best = val_acc
            [os.remove(''.join(['./checkpoint/', old_checkpoint])) for old_checkpoint in checkpoint_best]

    def save_config(self, args):
        try:
            with open('./checkpoint/last_config.py', 'w') as config_py:
                config_py.write('config = [\n' + '\n'.join('\t\'--{} {}\','
                    .format(arg, ' '.join(value) if isinstance(value, Iterable)
                    and not isinstance(value, str) else value)
                    for arg, value in vars(args).items()) + '\n]\n')
        except:
            print('Cannot save current config for checkpoint')
