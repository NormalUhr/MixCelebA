import csv
import random

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from metric import get_all_metrics


def write_all_csv(results, iter_name, column_name, file_name):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(([iter_name, column_name]))
        writer.writerows(results)


def write_csv(lists, iter_name, colmun_name, file_name):
    write_all_csv([(i, item) for i, item in enumerate(lists)], iter_name, colmun_name, file_name)


def write_csv_rows(file_name, column_list):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(column_list)


def get_one_hot(y, num_class, device):
    # Please check, y should start from 0 and be in range [0, num_class - 1]
    if len(y.shape) == 1:
        y_new = y.unsqueeze(-1)
    else:
        y_new = y
    y_one_hot = torch.FloatTensor(y_new.shape[0], num_class).to(device)
    y_one_hot.zero_()
    y_one_hot.scatter_(1, y_new, 1)
    return y_one_hot


def evaluation(test_loader, predictor, epoch, device):
    predictor.eval()
    pbar = tqdm(test_loader, total=len(test_loader), ncols=120, desc="Testing")
    fxs = []
    fxs_prob = []
    y_all = []
    d_all = []
    test_total_num = 0
    test_true_num = 0
    for x, (y, d) in pbar:
        y_all.append(y)
        d_all.append(d)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            lgt = predictor(x)
            fxs_prob.append(lgt)
        test_total_num += y.shape[0]
        pred = lgt.argmax(1)
        fxs.append(pred)
        test_true_num += (pred == y.view(-1)).type(torch.float).sum().detach().cpu().item()
        acc = test_true_num * 1.0 / test_total_num
        pbar.set_description(f"Test Epoch {epoch} Acc {100 * acc:.2f}%")
    pbar.set_description(f"Test Epoch {epoch} Acc {100 * test_true_num / test_total_num:.2f}%")
    y_all, d_all = torch.cat(y_all).view(-1).cpu().numpy(), torch.cat(d_all).view(-1).cpu().numpy()
    ds_dict = {"Male": d_all, "Female": 1 - d_all}
    fxs = torch.cat(fxs).view(-1).detach().cpu().numpy()
    fxs_prob = torch.cat(fxs_prob, dim=0).detach().cpu().numpy()
    ret_no_class_balance = get_all_metrics(y_true=y_all, y_pred=fxs, y_prob=fxs_prob, z=ds_dict,
                                           use_class_balance=False)
    return ret_no_class_balance, test_true_num / test_total_num


def get_transform(image_size):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
    ])

    return transform_train, transform_test


def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    torch.backends.cudnn.enabled = True

    # If you set the cudnn.benchmark the CuDNN library will benchmark several algorithms and pick that which it found to be fastest.
    # Rule of thumb: useful if you have fixed input sizes
    torch.backends.cudnn.benchmark = False

    # Some of the listed operations don't have a deterministic implementation. So if torch.use_deterministic_algorithms(True) is set, they will throw an error.
    torch.backends.cudnn.deterministic = True
