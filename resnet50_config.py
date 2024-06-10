resnet50_config = [
    # '--datasets',
    # 'output ./inference/output/predicts.txt',
    '--img_size 128',
    '--transform classic',

    '--num_workers 4',
    '--checkpoint resnet_72_20240610_classic_img156_b16_e3_checkpoint_best.pt',
    '--model resnet50',
    '--num-encoder-layers 3',
    '--emb-size 768',
    '--num-heads 4',
    '--batch_size 32',

    '--num_workers 12',
]