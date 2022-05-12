import argparse
import warnings

from torch.utils.data import DataLoader

from dataset import CelebABalance as CelebA
from utils import *



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="../data/celeba/celeba.hdf5")
    parser.add_argument('--domain-attrs', type=str, default='Male')
    parser.add_argument('--target-attrs', type=str, default='Blond_Hair')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--gr', type=float, default=0.0)
    parser.add_argument('--base-ratio', type=float, default=4)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    transform_train, transform_test = get_transform(image_size=224)

    train_set = CelebA(root=args.data_dir, target_attr=args.target_attrs,
                       transform=transform_test, split="train", gaussian_aug_ratio=args.gr, base_ratio=args.base_ratio)
    train_loader = DataLoader(train_set, batch_size=len(train_set), num_workers=args.num_workers, pin_memory=True)

    print(len(train_set))

    for x, (y, d) in train_loader:
        majority_var = torch.std(x[torch.where(y.view(-1) == 0)], dim=[2, -1])
        print(majority_var.mean())

        minority_var = torch.std(x[torch.where(y.view(-1) == 1)], dim=[2, -1])
        print(minority_var.mean())



