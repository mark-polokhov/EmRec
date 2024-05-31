'''
Config for resnet50
'''

# config = [
#     # '-a',
#     '--datasets affectnet_short',
#     '--img_size 128',
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

# config = [
#     # '-a',
#     '--datasets affectnet_short',
#     '--img_size 64',
#     '--transform classic',
#     '--val_split 0.1',
#     '--split_seed 18',

#     '--model vggtransformer',
#     '--num-encoder-layers 1',
#     '--num-decoder-layers 1',
#     '--emb-size 128',
#     '--num-heads 4',
#     '--epochs 50',
#     '--batch_size 4',
#     '--num_workers 12',
#     '--optimizer Adam',
#     '--save_every 1',
#     # '--checkpoint 20240227_e6_checkpoint_last.pt',
# ]

'''
Config for VisionTransformer
'''

config = [
    # '-a',
    '--datasets affectnet_short',
    '--img_size 128',
    '--transform classic',
    '--val_split 0.1',
    '--split_seed 18',

    '--model vit',
    '--num-encoder-layers 3',
    '--emb-size 768',
    '--num-heads 4',
    '--epochs 50',
    '--batch_size 32',
    '--num_workers 12',
    '--optimizer Adam',
    '--save_every 1',
    # '--checkpoint 20240519_classic_img128_b32_e10_checkpoint_best.pt',
]