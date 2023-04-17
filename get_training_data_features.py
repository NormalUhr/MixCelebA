import argparse
import warnings

import torch.nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18

from dataset import CelebAMultiBalance as CelebA
from models.model_zoo import *
from models.resnet9 import resnet9
from utils import *

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def pca_reduce(tensor, n_components):
    # Convert the tensor to a NumPy array
    array_data = tensor.numpy()

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(array_data)

    # Convert the reduced data back to a PyTorch tensor
    reduced_tensor = torch.tensor(reduced_data, dtype=torch.float32)

    return reduced_tensor

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
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--domain-attrs', type=str, default='Male')
    parser.add_argument('--target-attrs', type=str, nargs="+", default=[])
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18", "resnet20s", "resnet9"])
    parser.add_argument('--total-num', default=5000, type=int)
    parser.add_argument('--test-num', default=1000, type=int)

    parser.add_argument('--gv', type=float, default=0.05, help="Gaussian Noise")
    parser.add_argument('--gr', type=float, default=0.1)
    parser.add_argument('--minor-ratio', type=float, default=0.2, help="the ratio of minority group")
    parser.add_argument('--add-aug', type=str, default="gaussian", choices=["rotation", "crop", "gaussian"])

    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)

    args = parser.parse_args()

    assert args.domain_attrs in attr_list

    return args


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)

    # Sanity Check!
    assert args.data_dir is not None

    image_size = 224
    transform_train, transform_test = get_transform(image_size=image_size)

    num_attr = len(args.target_attrs)
    num_classes = 2 ** num_attr

    # init model
    if args.arch == "resnet18":
        predictor = resnet18(pretrained=False)
        predictor.fc = nn.Linear(512, num_classes)
    elif args.arch == "resnet9":
        predictor = resnet9(num_classes=num_classes)
    else:
        predictor = resnet20s(num_classes)
    predictor = predictor.to(device)

    feature_collector = []

    def feature_collector_hook(module, input):
        feature_collector.append(input[0].detach().cpu())

    predictor.fc.register_forward_pre_hook(feature_collector_hook)

    # Load checkpoints
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location=device)
        predictor.load_state_dict(checkpoint["predictor"])

    train_set = CelebA(root=args.data_dir, target_attrs=args.target_attrs, add_aug_ratio=args.gr,
                       num=args.total_num,
                       base_ratio=(1 - args.minor_ratio) / args.minor_ratio,
                       add_aug_mag=args.gv,
                       add_aug=args.add_aug,
                       transform=transform_train, split="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    predictor.eval()
    pbar = tqdm(train_loader, total=len(train_loader), ncols=120)
    with torch.no_grad():
        for x, (y, d) in pbar:
            x, y, d = x.to(device), y.to(device), d.to(device)

            predictor(x)

    feature_collector = torch.cat(feature_collector, dim=0)
    print(f"Collect features of the shape", feature_collector.shape)
    torch.save(feature_collector, args.save_path)

    # Fit a GMM with 2 components
    gm = GaussianMixture(n_components=2, random_state=42)
    gm.fit(feature_collector.numpy())

    # Compute the average log-likelihood score
    average_log_likelihood_score = gm.score(feature_collector.numpy())

    print(f"Average Log-Likelihood Score for Original Feature:", average_log_likelihood_score)

    for dim in [2, 5, 25, 100, 200, 300, 400]:
        # Perform PCA and reduce the dimensionality to 5
        reduced_data_tensor = pca_reduce(feature_collector, dim)
        torch.save(reduced_data_tensor, args.save_path.split(".")[0] + f"_pca{dim}" + args.save_path.split(".")[1])
        reduced_data_numpy = reduced_data_tensor.numpy()

        # Fit a GMM with 2 components
        gm = GaussianMixture(n_components=2, random_state=42)
        gm.fit(reduced_data_numpy)

        # Compute the average log-likelihood score
        average_log_likelihood_score = gm.score(reduced_data_numpy)

        print(f"Average Log-Likelihood Score for Dim {dim}:", average_log_likelihood_score)



if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
