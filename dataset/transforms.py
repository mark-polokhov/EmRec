import torchvision.transforms as transforms


def apply_transform(args):
    if args.transform == 'default':
        return default_transform(args)
    else:
        raise NotImplementedError


def default_transform(args):
    return transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])