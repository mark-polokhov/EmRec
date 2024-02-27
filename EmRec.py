#!/usr/bin/python
from config import config
from inference.config import infer_config

import subprocess
import argparse

parser = argparse.ArgumentParser(prog='EmRec')
parser.add_argument('--train', action='store_true',
                    help='Runs training of the EmRec')
parser.add_argument('--infer', action='store_true',
                    help='Runs infer of the EmRec')

args = parser.parse_args()


if __name__ == '__main__':
    if args.train:
        print('Starting training of the EmRec')
        print('Config:', ' '.join(config))
        subprocess.run(f'python train.py {" ".join(config)}', shell=True)
    elif args.infer:
        print('Starting inference of the EmRec')
        print('Config:', ' '.join(infer_config))
        subprocess.run(f'python infer.py {" ".join(infer_config)}', shell=True)
