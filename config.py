'''
Config for resnet50
'''

# config = [
#     # '-a',
#     '--datasets affectnet_short',
#     '--img_size 96',
#     '--transform classic',
#     '--val_split 0.1',
#     '--split_seed 18',

#     '--model resnet50',
#     '--epochs 50',
#     '--batch_size 32',
#     '--num_workers 12',
#     '--optimizer Adam',
#     '--save_every 1',
#     # '--checkpoint 20240227_e6_checkpoint_last.pt',
# ]

'''
Config for VGGTransformer
'''

config = [
    # '-a',
    '--datasets affectnet_short',
    '--img_size 72',
    '--transform classic',
    '--val_split 0.1',
    '--split_seed 18',

    '--model vggtransformer',
    '--num-encoder-layers 2',
    '--num-decoder-layers 2',
    '--num-heads 4',
    '--epochs 50',
    '--batch_size 8',
    '--num_workers 12',
    '--optimizer Adam',
    '--save_every 1',
    # '--checkpoint 20240227_e6_checkpoint_last.pt',
]
