import argparse
import warnings
import os
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18

from dataset import CelebABalance as CelebA
from models.model_zoo import *
from models.resnet9 import resnet9
from utils import *



def get_low_rank(matrix, rank=5):
    U, S, V = torch.svd(matrix)
    S[rank:] = 0

    low_matrix = torch.matmul(U, torch.diag(S))
    low_matrix = torch.matmul(low_matrix, V)

    return low_matrix


if __name__ == "__main__":

    matrix = torch.randn((1500, 256))
    low_matrix = get_low_rank(matrix, rank=10)

    print(low_matrix.shape)