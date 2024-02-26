import os
from PIL import Image
from torch import Generator
from torch.utils.data import Dataset, random_split


class MultiDataset(Dataset):
    def __init__(self, args, transform=None):
        super().__init__()

        self.labels_list = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        if args.all_datasets:
            self.datasets = [folder for folder in os.listdir('./dataset/')
                             if os.path.isdir(''.join(['./dataset/', folder]))
                             and not folder.lower().startswith('_')]
        else:
            self.datasets = args.datasets
        self.transform = transform

        print(self.datasets)

        self.images = []
        self.labels = []
        for dataset in self.datasets:
            for ind, label in enumerate(self.labels_list):
                dataset_path = ''.join(['./dataset/', dataset, '/', label])
                dataset_images = [''.join([dataset_path, '/', file])
                                  for file in os.listdir(dataset_path)
                                  if os.path.isfile(''.join([dataset_path, '/', file]))
                                  and file.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if dataset_images == []:
                    print("Dataset {} doesn't exist, skipping".format(dataset))
                    continue
                dataset_labels = [ind] * len(dataset_images)
                self.images.extend(dataset_images)
                self.labels.extend(dataset_labels)

    def __getitem__(self, ind):
        image = Image.open(self.images[ind])
        label = self.labels[ind]
        if self.transform is not None:
            return self.transform(image), label
        return image, label
        
    def __len__(self):
        return len(self.images)
    
    def inds2labels(self, inds):
        return [self.labels_list[ind] for ind in inds]

def train_val_split(dataset, args):
    if args.split_seed:
        generator = Generator().manual_seed(args.split_seed)
    else:
        generator = Generator()
    if args.val_split != None:
        return random_split(dataset, [1 - args.val_split, args.val_split], generator=generator)
    print('--val_split is not set, no validation will be proceeded')
    return dataset, Dataset()