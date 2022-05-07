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
    parser.add_argument('--gr', type=float, default=0.1)
    parser.add_argument('--base-ratio', type=float, default=0.25)

    args = parser.parse_args()

    return args


def main(args):
    image_size = 10

    transform_train, transform_test = get_transform(image_size=image_size)

    train_set = CelebA(root=args.data_dir, target_attr=args.target_attrs,
                       transform=transform_test, split="train", gaussian_aug_ratio=args.gr, base_ratio=args.base_ratio)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    print("===========================> Train Set <===========================")
    pbar = tqdm(train_loader, total=len(train_loader), ncols=120, desc="Testing")

    stat = {0: 0, 1: 1}
    stat_z = {0: 0, 1: 1}
    stat_cross = [0, 0, 0, 0]
    for x, (y, d) in pbar:
        for i in range(2):
            stat[i] += (y == i).sum().detach().item()
            stat_z[i] += (d == i).sum().detach().item()
            for j in range(2):
                stat_cross[i * 2 + j] += (((y == i) * (d == j)).sum().detach().item())
    print(args.target_attrs)
    print(stat)
    print(f"ratio: {stat[0] / (stat[0] + stat[1])} : {stat[1] / (stat[0] + stat[1])} = {stat[0] / stat[1]}")

    print("Male")
    print(stat_z)
    print(f"ratio: {stat_z[0] / (stat_z[0] + stat_z[1])} : {stat_z[1] / (stat_z[0] + stat_z[1])} = {stat_z[0] / stat_z[1]}")

    print("Cross")
    print(stat_cross)

    test_set = CelebA(root=args.data_dir, target_attr=args.target_attrs,
                      transform=transform_test, split="test", gaussian_aug_ratio=0.0, base_ratio=args.base_ratio)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    print("===========================> Test Set <===========================")

    pbar = tqdm(test_loader, total=len(test_loader), ncols=120, desc="Testing")
    stat = {0: 0, 1: 1}
    stat_z = {0: 0, 1: 1}
    stat_cross = [0, 0, 0, 0]
    for x, (y, d) in pbar:
        for i in range(2):
            stat[i] += (y == i).sum().detach().item()
            stat_z[i] += (d == i).sum().detach().item()
            for j in range(2):
                stat_cross[i * 2 + j] += (((y == i) * (d == j)).sum().detach().item())
    print(args.target_attrs)
    print(stat)
    print(f"ratio: {stat[0] / (stat[0] + stat[1])} : {stat[1] / (stat[0] + stat[1])} = {stat[0] / stat[1]}")

    print("Male")
    print(stat_z)
    print(
        f"ratio: {stat_z[0] / (stat_z[0] + stat_z[1])} : {stat_z[1] / (stat_z[0] + stat_z[1])} = {stat_z[0] / stat_z[1]}")

    print("Cross")
    print(stat_cross)


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
