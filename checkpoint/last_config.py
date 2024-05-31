config = [
	'--all_datasets False',
	'--datasets affectnet_short',
	'--img_size 128',
	'--transform classic',
	'--val_split 0.1',
	'--split_seed 18',
	'--epochs 50',
	'--batch_size 32',
	'--num_workers 12',
	'--checkpoint None',
	'--save_every 1',
	'--optimizer Adam',
	'--lr_scheduler None',
	'--model vit',
	'--num_encoder_layers 3',
	'--num_decoder_layers 3',
	'--num_heads 8',
	'--emb_size 768',
]
