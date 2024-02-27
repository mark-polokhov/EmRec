import torchvision.transforms as transforms


def apply_transform(args):
    if args.transform == 'default':
        return default_transform(args)
    elif args.transform == 'classic':
        return classic_transform(args)
    else:
        raise NotImplementedError


def default_transform(args):
    return transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

def classic_transform(args):
    return transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])