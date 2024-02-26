from dataset import MultiDataset, apply_transform, train_val_split
from trainer import Trainer

import config
import argparse
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(prog='EmRec')
# Dataset
parser.add_argument('-a', dest='all_datasets', action='store_true',
                    help='Use all available datasets') ###
parser.add_argument('-d', dest='datasets', type=str, nargs='+',
                    help='Datasets to use')
parser.add_argument('--img_size', type=int,
                    help='Every image in the Dataset will be transform to img_size')
parser.add_argument('--transform', type=str, default='default',
                    help='What tranform use for images in Dataset')
parser.add_argument('--val_split', type=float,
                    help='Fraction of validation set size (from 0 to 1)')
parser.add_argument('--split_seed', type=int,
                    help='Seed for dataset split for reproducibility')
# Other
parser.add_argument('-b', '--batch_size', type=int,
                    help='Batch size for training')
parser.add_argument('-j', '--num_workers', type=int, default=4,
                    help='Number of workers')
parser.add_argument('-ch', '--checkpoint', type=str,
                    help='name of the checkpoint file to start training with') ###
parser.add_argument('-s', '--save_every', type=int,
                    help='Make checkpoint after save_every number of epochs') ###
parser.add_argument('--optimizer', type=str,
                    help='Adam or SGD')
parser.add_argument('-m', '--model', type=str,
                    help='Which model to use (resnet50)') ###
parser.add_argument('-lr', '--lr_scheduler', type=str, default=None,
                    help='Which lr scheduler to use') ###
parser.add_argument('-e', '--epochs', type=int,
                    help='Number of epochs to train model')

args = parser.parse_args()


def run_training():
    dataset = MultiDataset(args, transform=apply_transform(args))
    train_dataset, val_dataset = train_val_split(dataset, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)
    val_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)
    trainer = Trainer(args)
    trainer.train(train_dataloader, val_dataloader, args)

if __name__ == '__main__':
    run_training()