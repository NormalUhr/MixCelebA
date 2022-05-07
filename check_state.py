import argparse
import warnings

from torch.utils.data import DataLoader

from dataset import CelebABalance as CelebA
from utils import *

warnings.filterwarnings("ignore")

attr_list = ('5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,'
             'Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,'
             'Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,'
             'Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,'
             'Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young'
             ).split(',')

attr_dict = {}
for i, attr in enumerate(attr_list):
    attr_dict[attr] = i

insufficient_attr_list = '5_o_Clock_Shadow,Goatee,Mustache,Sideburns,Wearing_Necktie'.split(',')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="../data/celeba/celeba.hdf5")
    parser.add_argument('--domain-attrs', type=str, default='Male')
    parser.add_argument('--target-attrs', type=str, default='Blond_Hair')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--method', '--m', type=str, default="std", choices=['std', 'adv', 'repro', 'rpatch', 'roptim'],
                        help="Method: standard training, adv training, reprogram (vanilla, patch, optimization-based)")

    args = parser.parse_args()

    return args


def main(args):
    image_size = 10

    transform_train, transform_test = get_transform(image_size=image_size)

    train_set = CelebA(root=args.data_dir, target_attr=args.target_attrs,
                       transform=transform_test, split="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    pbar = tqdm(train_loader, total=len(train_loader), ncols=120, desc="Testing")

    print("===========================> Train Set <===========================")
    stat = {0: 0, 1: 1}
    for x, (y, d) in pbar:
        for i in range(2):
            stat[i] += (y == i).sum().detach().item()
    print(args.target_attrs)
    print(stat)

    print(f"ratio: {stat[0] / (stat[0] + stat[1])} : {stat[1] / (stat[0] + stat[1])} = {stat[0] / stat[1]}")

    test_set = CelebA(root=args.data_dir, target_attr=args.target_attrs,
                      transform=transform_test, split="test")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    pbar = tqdm(test_loader, total=len(test_loader), ncols=120, desc="Testing")

    print("===========================> Test Set <===========================")
    stat = {0: 0, 1: 1}
    for x, (y, d) in pbar:
        for i in range(2):
            stat[i] += (y == i).sum().detach().item()
    print(args.target_attrs)
    print(stat)

    print(f"ratio: {stat[0] / (stat[0] + stat[1])} : {stat[1] / (stat[0] + stat[1])} = {stat[0] / stat[1]}")


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)