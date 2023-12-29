import config
import argparse

from dataset import MultiDataset


parser = argparse.ArgumentParser(prog='Dataset')
parser.add_argument('-a', dest='all_datasets', action='store_true',
                    help='Use all available datasets')
parser.add_argument('-d', dest='datasets', type=str, nargs='+',
                    help='Datasets to use')

args = parser.parse_args()


def init_training():
    dataset = MultiDataset(args)
    print(dataset[0])


if __name__ == '__main__':
    init_training()