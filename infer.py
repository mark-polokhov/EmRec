from dataset import MultiDataset, apply_transform, inds2labels
from inference import Inferencer

import argparse
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(prog='EmRec [Inference]')
# Dataset
parser.add_argument('-a', dest='all_datasets', action='store_true',
                    help='Use all available datasets') ###
parser.add_argument('-d', '--datasets', type=str, nargs='+',
                    help='Datasets to use')
parser.add_argument('--img_size', type=int,
                    help='Every image in the Dataset will be transform to img_size')
parser.add_argument('--transform', type=str, default='default',
                    help='What tranform use for images in Dataset')
# Other
parser.add_argument('-j', '--num_workers', type=int, default=4,
                    help='Number of workers')
parser.add_argument('-ch', '--checkpoint', type=str,
                    help='name of the checkpoint file to start training with') ###
parser.add_argument('-m', '--model', type=str,
                    help='Which model to use (resnet50, vggtransformer)')
parser.add_argument('--num-encoder-layers', type=int, default=3,
                    help='Number of encoder layers for VGGTransformer')
parser.add_argument('--num-decoder-layers', type=int, default=3,
                    help='Number of decoder layers for VGGTransformer')
parser.add_argument('--num-heads', type=int, default=8,
                    help='Number of heads for VGGTransformer')
parser.add_argument('--vgg-emb-size', type=int, default=8,
                    help='Embedding size for VGGTransformer')

args = parser.parse_args()


def write_predicts(images, predicts):
    try:
        with open('./inference/output/predicts.txt', 'w') as output:
            output.write('\n'.join(list(map(' '.join, list(zip(images, inds2labels([pred.item() for pred in predicts])))))))
        print('Inference ended successfully')
    except:
        print('Cannot save inference result into \'./inference/output/predicts.txt\', aborting')

def run_inference():
    dataset = MultiDataset(args, train=False, transform=apply_transform(args))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)
    inferencer = Inferencer(args)
    predicts = inferencer.inference(dataloader)
    write_predicts(dataset.get_images(), predicts)

if __name__ == "__main__":
    run_inference()