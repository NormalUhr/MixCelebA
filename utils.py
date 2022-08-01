import csv
import random

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

import torch.nn as nn


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
    test_total_num = 0
    test_true_num = 0
    test_total_man = 0
    test_total_woman = 0
    test_true_man = 0
    test_true_woman = 0
    test_total_loss = 0.0
    test_loss_man = 0.0
    test_loss_woman = 0.0
    for x, (y, d) in pbar:
        x = x.to(device)
        y = y.to(device)
        d = d.to(device)
        y_one_hot = get_one_hot(y, 2, device)
        y_man_one_hot = get_one_hot(y[d == 1], 2, device)
        y_woman_one_hot = get_one_hot(y[d == 0], 2, device)
        with torch.no_grad():
            lgt = predictor(x)
            lgt_man = predictor(x[d == 1])
            lgt_woman = predictor(x[d == 0])
        test_total_num += y.shape[0]
        test_total_man += (d == 1).type(torch.float).sum().detach().cpu().item()
        test_total_woman += (d == 0).type(torch.float).sum().detach().cpu().item()
        pred = lgt.argmax(1)
        test_total_loss += (nn.functional.cross_entropy(lgt, y_one_hot) * y.shape[0]).detach().cpu().item()
        test_loss_man += (nn.functional.cross_entropy(lgt_man, y_man_one_hot) * y[d == 1].shape[0]).detach().cpu().item()
        test_loss_woman += (nn.functional.cross_entropy(lgt_woman, y_woman_one_hot) * y[d == 0].shape[0]).detach().cpu().item()
        test_true_num += (pred == y.view(-1)).type(torch.float).sum().detach().cpu().item()
        test_true_man += ((pred == y.view(-1)).view(-1) * (d == 1).view(-1)).type(torch.float).sum().detach().cpu().item()
        test_true_woman += ((pred == y.view(-1)).view(-1) * (d == 0).view(-1)).type(torch.float).sum().detach().cpu().item()
        acc = test_true_num * 1.0 / test_total_num
        pbar.set_description(f"Test Epoch {epoch} Acc {100 * acc:.2f}%")
    pbar.set_description(f"Test Epoch {epoch} Acc {100 * test_true_num / test_total_num:.2f}%")

    return test_true_num / test_total_num, test_true_man / test_total_man, test_true_woman / test_total_woman, test_total_loss / test_total_num, test_loss_man / test_total_man, test_loss_woman / test_total_woman


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
