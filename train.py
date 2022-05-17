import argparse
import os
import time
import sys
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from torch.cuda.amp import autocast, GradScaler
from dataset import CelebABalance as CelebA
from models.model_zoo import *
from models.resnet9 import resnet9
from utils import *
import warnings
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
    parser.add_argument('--checkpoint', type=str, default=None)
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
    parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18", "resnet20s", "resnet9"])
    parser.add_argument('--evaluate', action="store_true")
    parser.add_argument('--total-num', default=None, type=int)

    parser.add_argument('--gv', type=float, default=0.05, help="Gaussian Noise")
    parser.add_argument('--gr', type=float, default=0.1)
    parser.add_argument('--minor-ratio', type=float, default=0.2, help="the ratio of minority group")

    args = parser.parse_args()

    assert args.target_attrs in attr_list
    assert args.domain_attrs in attr_list

    return args


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)

    # Sanity Check!
    assert args.data_dir is not None

    # make save path dir
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "csv"), exist_ok=True)
    model_attr_name = args.arch + "_" + "_target_"
    model_attr_name += str(attr_dict[args.target_attrs]) + "_"
    model_attr_name += f'seed{args.seed}'
    model_attr_name += f'_gr{args.gr}_gv{args.gv}'
    if args.total_num is not None:
        model_attr_name += f"_num{args.total_num}"
    if args.exp_name is not None:
        model_attr_name += f'_{args.exp_name}'

    image_size = 224
    transform_train, transform_test = get_transform(image_size=image_size)

    num_class = 2

    # init model
    if args.arch == "resnet18":
        predictor = resnet18(pretrained=False)
        predictor.fc = nn.Linear(512, num_class)
    elif args.arch == "resnet9":
        predictor = resnet9(num_classes=num_class)
    else:
        predictor = resnet20s(num_class)
    predictor = predictor.to(device)
    p_optim = torch.optim.Adam(predictor.parameters(), lr=args.lr, weight_decay=args.wd)
    p_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(p_optim,
                                                          gamma=0.1,
                                                          milestones=[int(0.8 * args.epochs),
                                                                      int(0.9 * args.epochs)])

    # Load checkpoints
    best_SA = 0.0
    acc_best_man = 0.0
    acc_best_woman = 0.0
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        predictor.load_state_dict(checkpoint["predictor"])
        if args.resume:
            p_optim.load_state_dict(checkpoint["p_optim"])
            p_lr_scheduler.load_state_dict(checkpoint["p_lr_scheduler"])
            best_SA = checkpoint["best_SA"]
            acc_best_man = checkpoint["acc_best_man"]
            acc_best_woman = checkpoint["acc_best_woman"]
            start_epoch = checkpoint["epoch"]
    test_set = CelebA(root=args.data_dir, target_attr=args.target_attrs,
                      transform=transform_test, split="test", gaussian_aug_ratio=0.0, base_ratio=1.0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    if args.evaluate:
        print("================= Evaluating on Test Set before Training =================")
        accuracy, acc_man, acc_woman = evaluation(test_loader, predictor, -1, device)
        print("The accuracy is {:.4f}, {:.4f}, {:.4f}".format(accuracy, acc_man, acc_woman))
        if args.evaluate:
            sys.exit()

    train_set = CelebA(root=args.data_dir, target_attr=args.target_attrs, gaussian_aug_ratio=args.gr,
                       num=args.total_num,
                       base_ratio=(1-args.minor_ratio) / args.minor_ratio,
                       gaussian_variance=args.gv,
                       transform=transform_train, split="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    scaler = GradScaler()

    for epoch in range(start_epoch, args.epochs):
        # training
        predictor.train()
        end = time.time()
        print(f"======================================= Epoch {epoch} =======================================")
        pbar = tqdm(train_loader, total=len(train_loader), ncols=120)
        total_num = 0
        true_num = 0
        for x, (y, d) in pbar:
            x, y, d = x.to(device), y.to(device), d.to(device)
            y_one_hot = get_one_hot(y, num_class, device)  # one-hot [bs, num_class]
            p_optim.zero_grad()

            with autocast():
                lgt = predictor(x)
                pred_loss = nn.functional.cross_entropy(lgt, y_one_hot)
                scaler.scale(pred_loss).backward()
                scaler.step(p_optim)
                scaler.update()

            # results for this batch
            total_num += y.size(0)
            true_num += (lgt.argmax(1) == y.view(-1)).type(torch.float).sum().detach().cpu().item()
            acc = true_num * 1.0 / total_num
            pbar.set_description(f"Training Epoch {epoch} Acc {100 * acc:.4f}%")
        pbar.set_description(f"Training Epoch {epoch} Acc {100 * true_num / total_num:.4f}%")

        p_lr_scheduler.step()

        print("================= Test Set =================")
        accuracy, acc_man, acc_woman = evaluation(test_loader, predictor, epoch, device)

        metric = accuracy
        if metric > best_SA:
            print("+++++++++++ Find New Best Min ACC +++++++++++")
            best_SA = metric
            acc_best_man = acc_man
            acc_best_woman = acc_woman
            cp = {"predictor": predictor.state_dict(),
                  "p_optim": p_optim.state_dict(),
                  "p_lr_scheduler": p_lr_scheduler.state_dict(),
                  "epoch": epoch,
                  "best_SA": best_SA,
                  "acc_best_man": acc_best_man,
                  "acc_best_woman": acc_best_woman
                  }
            torch.save(cp,
                       os.path.join(os.path.join(args.result_dir, "checkpoints"), f'{model_attr_name}_best.pth.tar'))

        print("The acc is {:.4f}, {:.4f}, {:.4f}".format(best_SA, acc_best_man, acc_best_woman))

        print(f"Time Consumption for one epoch is {time.time() - end}s")

    print("The final acc is {:.4f}, {:.4f}, {:.4f}".format(best_SA, acc_best_man, acc_best_woman))


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
