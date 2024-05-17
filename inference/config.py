infer_config = [
    '--datasets video_example',
    '--img_size 64',
    '--transform classic',

    '--vgg-emb-size 128',
    '--num_workers 4',
    '--checkpoint 20240509_classic_img64_b4_e1_checkpoint_best.pt',
    '--model vggtransformer',
    '--num-encoder-layers 1',
    '--num-decoder-layers 1',
    '--num-heads 4',
]