config = [
	'--all_datasets False',
	'--datasets affectnet_short',
	'--img_size 96',
	'--transform classic',
	'--val_split 0.1',
	'--split_seed 18',
	'--epochs 50',
	'--batch_size 32',
	'--num_workers 12',
	'--checkpoint None',
	'--save_every 1',
	'--optimizer Adam',
	'--model resnet50',
	'--lr_scheduler None',
]
