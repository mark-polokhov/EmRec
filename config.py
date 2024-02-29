config = [
    # '-a',
    '--datasets affectnet_short fer2013',
    '--img_size 156',
    '--transform classic',
    '--val_split 0.1',
    '--split_seed 18',

    '--model resnet50',
    '--epochs 50',
    '--batch_size 32',
    '--num_workers 12',
    '--optimizer Adam',
    '--save_every 1',
    # '--checkpoint 20240227_e6_checkpoint_last.pt',
]