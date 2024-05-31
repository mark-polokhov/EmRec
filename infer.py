from dataset import MultiDataset, apply_transform, inds2labels
from inference import Inferencer

import argparse
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(prog='EmRec [Inference]')
# Dataset
parser.add_argument('-d', '--datasets', type=str, nargs='+',
                    help='Datasets to use')
parser.add_argument('--img_size', type=int,
                    help='Every image in the Dataset will be transform to img_size')
parser.add_argument('--transform', type=str, default='default',
                    help='What tranform use for images in Dataset')
parser.add_argument('--output', '-o', type=str,
                    help='Where to save inference results')
# Other
parser.add_argument('-j', '--num_workers', type=int, default=4,
                    help='Number of workers')
parser.add_argument('-ch', '--checkpoint', type=str,
                    help='name of the checkpoint file to start training with') ###
parser.add_argument('-m', '--model', type=str,
                    help='Which model to use (resnet50, vggtransformer)')
parser.add_argument('-b', '--batch_size', type=int,
                    help='Batch size for training')
# parser.add_argument('--num-encoder-layers', type=int, default=3,
#                     help='Number of encoder layers for VGGTransformer')
# parser.add_argument('--num-decoder-layers', type=int, default=3,
#                     help='Number of decoder layers for VGGTransformer')
# parser.add_argument('--num-heads', type=int, default=8,
#                     help='Number of heads for VGGTransformer')
# parser.add_argument('--vgg-emb-size', type=int, default=8,
#                     help='Embedding size for VGGTransformer')

# VisionTransformer
parser.add_argument('--num-encoder-layers', type=int, default=3,
                    help='Number of encoder layers for VGGTransformer')
parser.add_argument('--num-decoder-layers', type=int, default=3,
                    help='Number of decoder layers for VGGTransformer')
parser.add_argument('--num-heads', type=int, default=8,
                    help='Number of heads for VGGTransformer')
parser.add_argument('--emb-size', type=int, default=8,
                    help='Embedding size for Transformers')

args = parser.parse_args()


def write_predicts(predicts, classes, uuid=None):
    try:
        with open(args.output, 'w') as output:
            output.write(' '.join(classes) + '\n')
            predicts_printable = [predicts_row.tolist() for predicts_row in predicts]
            output.write('\n'.join([' '.join(map(str, map(lambda x: round(x, 3), row))) for row in predicts_printable]))
            # output.write('\n'.join(list(map(' '.join, list(zip(inds2labels([pred.item() for pred in pred_classes])))))))
        print('Inference ended successfully')
    except Exception as e:
        print(e)
        print('Cannot save inference result into \'./inference/output/predicts.txt\', aborting')

def smooth_predicts(predicts):
    pass

def run_inference(uuid=None):
    dataset = MultiDataset(args, train=False, transform=apply_transform(args))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    inferencer = Inferencer(args)
    # pred_classes, predicts = inferencer.inference(dataloader)
    predicts = inferencer.inference(dataloader)
    # write_predicts(dataset.get_images(), predicts)
    write_predicts(predicts, dataset.labels_list, uuid)

if __name__ == "__main__":
    run_inference()