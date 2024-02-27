from models import ConvLayers

import torch
from tqdm import tqdm


class Inferencer():
    def __init__(self, args):
        if args.model.lower() == 'resnet50':
            self.model = ConvLayers()
        else:
            raise NotImplementedError
        
        if args.checkpoint:
            self.model.load_state_dict(torch.load(''.join(['./checkpoint/', args.checkpoint])))

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('Currently using device:', self.device)

        self.model.to(self.device)

    def inference(self, dataloader):
        self.model.eval()

        predicts = []
        with torch.no_grad():
            for images in tqdm(dataloader):
                images = images.to(self.device)

                logits = self.model(images)
                _, predicted = torch.max(logits.data, 1)
                predicts.extend(predicted)

        return predicts
