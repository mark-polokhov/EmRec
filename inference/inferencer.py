from models import ConvLayers, VGGTransformer

import torch
from tqdm import tqdm


class Inferencer():
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
                                        emb_size=args.vgg_emb_size, nhead=args.num_heads)
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

                if self.model_name == 'resnet50':
                    logits = self.model(images)
                elif self.model_name == 'vggtransformer':
                    logits = self.model(images, torch.full((images.shape[0],), 1).to(self.device))
                _, predicted = torch.max(logits.data, 1)
                predicts.extend(predicted)

        return predicts
