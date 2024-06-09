vit_config = [
    # '--datasets',
    # 'output ./inference/output/predicts.txt',
    '--img_size 128',
    '--transform classic',

    '--num_workers 4',
    '--checkpoint vit_best_20240519_classic_img128_b32_e8_checkpoint_best.pt',
    '--model vit',
    '--num-encoder-layers 3',
    '--emb-size 768',
    '--num-heads 4',
    '--batch_size 32',

    '--num_workers 12',
]