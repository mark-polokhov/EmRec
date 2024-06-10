from models import ConvLayers, VGGTransformer, ViT

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
        elif self.model_name in {'vit', 'visiontransformer'}:
            self.model = ViT(img_size=args.img_size,
                            num_layers=args.num_encoder_layers,
                            num_heads=args.num_heads,
                            emb_size=args.emb_size)
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
        # pred_class = []
        with torch.no_grad():
            for images in tqdm(dataloader):
                images = images.to(self.device)

                if self.model_name == 'resnet50':
                    logits = self.model(images)
                elif self.model_name == 'vggtransformer':
                    logits = self.model(images, torch.full((images.shape[0],), 1).to(self.device))
                elif self.model_name in {'vit', 'visiontransformer'}:
                    logits = self.model(images).logits
                preds = torch.nn.functional.softmax(logits, dim=1)
                # _, predicted = torch.max(preds.data, 1)
                # pred_class.extend(predicted)
                predicts.extend(preds)

        # return pred_class, preds
        return predicts
