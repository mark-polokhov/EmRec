config = [
    # '-a',
    '--datasets affectnet_short',
    '--img_size 128',
    '--transform default',
    '--val_split 0.1',
    '--split_seed 18',

    '--epochs 5',
    '--batch_size 32',
    '--num_workers 4',
    '--optimizer Adam',
    '--save_every 1',
    '--checkpoint 20240226_e4_checkpoint_last.pt',
]