import config
import argparse

from dataset import MultiDataset, apply_transform


parser = argparse.ArgumentParser(prog='EmRec')
# Dataset
parser.add_argument('-a', dest='all_datasets', action='store_true',
                    help='Use all available datasets')
parser.add_argument('-d', dest='datasets', type=str, nargs='+',
                    help='Datasets to use')
parser.add_argument('--img_size', type=int, default=128,
                    help='Every image in the Dataset will be transform to img_size')
parser.add_argument('--transform', type=str, default='default',
                    help='What tranform use for images in Dataset')

args = parser.parse_args()


def init_training():
    dataset = MultiDataset(args, transform=apply_transform(args))
    print(dataset[0])


if __name__ == '__main__':
    init_training()