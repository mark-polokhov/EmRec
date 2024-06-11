vit_config = [
    # '--datasets',
    # 'output ./inference/output/predicts.txt',
    '--img_size 224',
    '--transform classic',

    '--num_workers 4',
    '--checkpoint vit_79_20240610_classic_img224_b64_e6_checkpoint_best.pt',
    '--model vit',
    '--num-encoder-layers 3',
    '--emb-size 768',
    '--num-heads 4',
    '--batch_size 64',

    '--num_workers 12',
]