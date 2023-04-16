import argparse
import os
import warnings

from torch.utils.data import DataLoader

from dataset import CelebABalance as CelebA
from models.resnet9 import resnet9
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
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--result-dir', type=str, default='results')
    parser.add_argument('--checkpoint', type=str, default=None, required=True)
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--domain-attrs', type=str, default='Male')
    parser.add_argument('--target-attrs', type=str, default='High_Cheekbones')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--arch', type=str, default="resnet9", choices=["resnet20s", "resnet9"])
    parser.add_argument('--evaluate', action="store_true")
    parser.add_argument('--total-num', default=5000, type=int)
    parser.add_argument('--test-num', default=1000, type=int)

    parser.add_argument('--gv', type=float, default=0.05, help="Gaussian Noise")
    parser.add_argument('--gr', type=float, default=0.1)
    parser.add_argument('--minor-ratio', type=float, default=0.2, help="the ratio of minority group")
    parser.add_argument('--add-aug', type=str, default="rotation", choices=["rotation", "crop", "gaussian"])

    args = parser.parse_args()

    assert args.target_attrs in attr_list
    assert args.domain_attrs in attr_list

    return args


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)

    # Sanity Check!
    assert args.data_dir is not None

    image_size = 224
    transform_train, transform_test = get_transform(image_size=image_size)

    num_class = 2

    # init model
    predictor = resnet9(num_class).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    predictor.load_state_dict(checkpoint["predictor"])
    test_set = CelebA(root=args.data_dir, target_attr=args.target_attrs,
                      transform=transform_test, split="test", add_aug_ratio=0.0, base_ratio=1.0,
                      num=args.test_num)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    print("================= Evaluating on Test Set before Training =================")

    predictor.eval()
    pbar = tqdm(test_loader, total=len(test_loader), ncols=120, desc="Testing")
    test_total_num = 0
    test_true_num = 0
    test_total_man = 0
    test_total_woman = 0
    test_true_man = 0
    test_true_woman = 0
    test_loss_man = 0.0
    test_loss_woman = 0.0
    total_features = []
    male_features = []
    female_features = []
    total_label = None
    total_domain = None
    for x, (y, d) in pbar:
        x = x.to(device)
        y = y.to(device)
        d = d.to(device)
        total_label = y
        total_domain = d
        y_man_one_hot = get_one_hot(y[d == 1], 2, device)
        y_woman_one_hot = get_one_hot(y[d == 0], 2, device)
        with torch.no_grad():
            lgt, features = predictor(x, True)
            lgt_man, man_features = predictor(x[d == 1], True)
            lgt_woman, woman_features = predictor(x[d == 0], True)
            total_features.append(features)
            male_features.append(man_features)
            female_features.append(woman_features)
        test_total_num += y.shape[0]
        test_total_man += (d == 1).type(torch.float).sum().detach().cpu().item()
        test_total_woman += (d == 0).type(torch.float).sum().detach().cpu().item()
        pred = lgt.argmax(1)
        if y[d == 1].shape[0] > 0:
            test_loss_man += (
                    nn.functional.cross_entropy(lgt_man, y_man_one_hot) * y[d == 1].shape[0]).detach().cpu().item()
        if y[d == 0].shape[0] > 0:
            test_loss_woman += (nn.functional.cross_entropy(lgt_woman, y_woman_one_hot) * y[d == 0].shape[
                0]).detach().cpu().item()
        test_true_num += (pred == y.view(-1)).type(torch.float).sum().detach().cpu().item()
        test_true_man += ((pred == y.view(-1)).view(-1) * (d == 1).view(-1)).type(
            torch.float).sum().detach().cpu().item()
        test_true_woman += ((pred == y.view(-1)).view(-1) * (d == 0).view(-1)).type(
            torch.float).sum().detach().cpu().item()
        acc = test_true_num * 1.0 / test_total_num
        pbar.set_description(f"Test Epoch Acc {100 * acc:.2f}%")
    pbar.set_description(f"Test Epoch Acc {100 * test_true_num / test_total_num:.2f}%")

    total_features = torch.stack(total_features).squeeze(dim=0)
    male_features = torch.stack(male_features).squeeze(dim=0)
    female_features = torch.stack(female_features).squeeze(dim=0)

    total_num = features.shape[0]
    male_num = male_features.shape[0]
    female_num = female_features.shape[0]
    print(f"male number: {male_num}")
    print(f"female number: {female_num}")

    mean_total_features = torch.mean(total_features, dim=0)
    mean_male_features = torch.mean(male_features, dim=0)
    mean_female_features = torch.mean(female_features, dim=0)

    offset_total_features = total_features - mean_total_features
    print(offset_total_features.shape)
    offset_male_features = male_features - mean_male_features
    offset_female_features = female_features - mean_female_features

    covariance_matrix = torch.matmul(offset_total_features.t(), offset_total_features) / total_num
    print(covariance_matrix.shape)
    covariance_male_matrix = torch.matmul(offset_male_features.t(), offset_male_features) / male_num
    covariance_female_matrix = torch.matmul(offset_female_features.t(), offset_female_features) / female_num

    total_U, total_S, total_V = torch.svd(covariance_matrix)
    total_ratio = total_S.max() / total_S.min()
    print(total_S.shape)
    male_U, male_S, male_V = torch.svd(covariance_male_matrix)
    male_ratio = male_S.max() / male_S.min()
    female_U, female_S, female_V = torch.svd(covariance_female_matrix)
    female_ratio = female_S.max() / female_S.min()

    print(total_ratio.cpu().item(), male_ratio.cpu().item(), female_ratio.cpu().item())
    print(total_S.max().cpu().item(), male_S.max().cpu().item(), female_S.max().cpu().item())
    print(total_S.min().cpu().item(), male_S.min().cpu().item(), female_S.min().cpu().item())

    for i in range(total_S.shape[0]):
        print(f"{i}: {total_S[i].cpu().item()}, {male_S[i].cpu().item()}, {female_S[i].cpu().item()}")

    torch.save(total_S, os.path.join(args.result_dir, "singular_total.pth"))
    torch.save(male_S, os.path.join(args.result_dir, "singular_male.pth"))
    torch.save(female_S, os.path.join(args.result_dir, "singular_female.pth"))
    accuracy, acc_man, acc_woman = test_true_num / test_total_num, test_true_man / test_total_man, test_true_woman / test_total_woman
    print("The accuracy is {:.4f}, {:.4f}, {:.4f}".format(accuracy, acc_man, acc_woman))

    index = 5

    total_S[index:] = 0
    male_S[index:] = 0
    female_S[index:] = 0

    total_low_features = get_low_rank(total_features)
    # male_low_features = get_low_rank(male_features)
    # female_low_features = get_low_rank(female_features)

    total_pred_low_rank = predictor.linear(total_low_features).argmax(1)
    # male_pred_low_rank = predictor.linear(male_low_features).argmax(1)
    # female_pred_low_rank = predictor.linear(female_low_features).argmax(1)

    low_test_true_num = (total_pred_low_rank == total_label.view(-1)).type(torch.float).sum().detach().cpu().item()
    low_test_true_man = ((total_pred_low_rank == total_label.view(-1)).view(-1) * (total_domain == 1).view(-1)).type(
        torch.float).sum().detach().cpu().item()
    low_test_true_woman = ((total_pred_low_rank == total_label.view(-1)).view(-1) * (total_domain == 0).view(-1)).type(
        torch.float).sum().detach().cpu().item()

    accuracy, acc_man, acc_woman = low_test_true_num / test_total_num, low_test_true_man / test_total_man, low_test_true_woman / test_total_woman

    print("Low Rank: The accuracy is {:.4f}, {:.4f}, {:.4f}".format(accuracy, acc_man, acc_woman))


def get_low_rank(matrix, rank=5):
    U, S, V = torch.svd(matrix)
    S[rank:] = 0

    low_matrix = torch.matmul(U, torch.diag(S))
    low_matrix = torch.matmul(low_matrix, V)

    return low_matrix


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
