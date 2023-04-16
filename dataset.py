import os
import random

import h5py
import numpy as np
import pandas
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, Resize, RandomRotation, Compose
from torchvision.transforms import ToTensor
from tqdm import tqdm

attr_list = ('5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,'
             'Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,'
             'Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,'
             'Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,'
             'Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young'
             ).split(',')


class CelebA(Dataset):
    def __init__(self, root_dir, target_attrs, domain_attrs=None, img_transform=ToTensor(), type="train") -> None:
        super().__init__()
        self.type = type
        self.img_dir = os.path.join(root_dir, 'img_align_celeba_png')
        self.table = self.__load_table(os.path.join(root_dir, 'list_attr_celeba.csv'))
        self.target_attrs = target_attrs
        self.domain_attrs = domain_attrs
        self.img_transform = img_transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.table.iloc[index, 0]))
        if self.img_transform is not None:
            img = self.img_transform(img)
        labels = self.table[self.target_attrs].iloc[index] if isinstance(self.target_attrs, str) else \
            self.table[self.target_attrs].iloc[index].to_numpy()
        if self.domain_attrs is not None:
            domains = self.table[self.domain_attrs].iloc[index] if isinstance(self.domain_attrs, str) else \
                self.table[self.domain_attrs].iloc[index].to_numpy()
            return img, (labels, domains)
        else:
            return img, labels

    def __load_table(self, path):
        whole_table = pandas.read_csv(path)
        train_point = 162770
        val_point = 182637
        if self.type == "train":
            return whole_table.iloc[:train_point]
        elif self.type == "val":
            return whole_table.iloc[train_point:val_point]
        elif self.type == "test":
            return whole_table.iloc[val_point:]
        else:
            raise ValueError("Invalid dataset type!")


class CelebATrigger(Dataset):
    def __init__(self, root_dir, target_attrs, domain_attrs=None, img_transform=ToTensor(), type="train",
                 trigger_data_num=0) -> None:
        super().__init__()
        self.type = type
        self.trigger_data = trigger_data_num
        self.img_dir = os.path.join(root_dir, 'img_align_celeba_png')
        self.table = self.__load_table(os.path.join(root_dir, 'list_attr_celeba.csv'))
        self.target_attrs = target_attrs
        self.domain_attrs = domain_attrs
        self.img_transform = img_transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.table.iloc[index, 0]))
        if self.img_transform is not None:
            img = self.img_transform(img)
        labels = self.table[self.target_attrs].iloc[index] if isinstance(self.target_attrs, str) else \
            self.table[self.target_attrs].iloc[index].to_numpy()
        if self.domain_attrs is not None:
            domains = self.table[self.domain_attrs].iloc[index] if isinstance(self.domain_attrs, str) else \
                self.table[self.domain_attrs].iloc[index].to_numpy()
            return img, (labels, domains)
        else:
            return img, labels

    def __load_table(self, path):
        whole_table = pandas.read_csv(path)
        train_point = 162770
        val_point = 182637
        if self.type == "train":
            return whole_table.iloc[:train_point - self.trigger_data]
        elif self.type == "trigger":
            return whole_table.iloc[train_point - self.trigger_data: train_point]
        elif self.type == "val":
            return whole_table.iloc[train_point:val_point]
        elif self.type == "test":
            return whole_table.iloc[val_point:]
        else:
            raise ValueError("Invalid dataset type!")

    def compute_balancing_class_weight(self):  # Multiple Domain Not Supported
        target = self.table[self.target_attrs].to_numpy()
        domain_label = self.table[self.domain_attrs].to_numpy()
        per_class_weight = []
        for i in range(target.shape[1]):
            class_label = target[:, i]
            cp = class_label.sum()  # class is positive
            cn = target.shape[0] - cp  # class is negative
            cn_dn = ((class_label + domain_label) == 0).sum()  # class is negative, domain is negative
            cn_dp = ((class_label - domain_label) == -1).sum()
            cp_dn = ((class_label - domain_label) == 1).sum()
            cp_dp = ((class_label + domain_label) == 2).sum()

            per_class_weight.append(
                (class_label * cp + (1 - class_label) * cn) /
                (2 * (
                        (1 - class_label) * (1 - domain_label) * cn_dn
                        + (1 - class_label) * domain_label * cn_dp
                        + class_label * (1 - domain_label) * cp_dn
                        + class_label * domain_label * cp_dp
                )
                 )
            )
        return per_class_weight


class CelebAFast(Dataset):
    def __init__(self, root, target_attrs, domain_attrs=None, img_transform=None, type="train",
                 trigger_data_num=0) -> None:
        super().__init__()
        assert type in ["train", "val", "test", "trigger"]
        self.type = type
        # self.true_type only in  ["train", "val", "test"]
        self.true_type = type if type != "trigger" else "train"
        self.root = root
        self.img_transform = img_transform
        self.trigger_data_num = trigger_data_num
        if isinstance(target_attrs, str):
            self.target_attrs = [bytes(target_attrs, 'utf-8')]
        else:
            self.target_attrs = [bytes(target_attr, 'utf-8') for target_attr in target_attrs]
        if domain_attrs is not None:
            if isinstance(domain_attrs, str):
                self.domain_attrs = [bytes(domain_attrs, 'utf-8')]
            else:
                self.domain_attrs = [bytes(domain_attr, 'utf-8') for domain_attr in domain_attrs]
        else:
            self.domain_attrs = None

        if isinstance(target_attrs, list):
            self.num_classes = 2 ** len(self.target_attrs)
        else:
            self.num_classes = 2

        self.labels = []
        self.y_index = []
        self.z_index = []
        with h5py.File(self.root, mode='r') as file:
            if isinstance(np.array(file["columns"])[0], str):
                # Sometimes np.array(file["columns"])[0] is bytes and sometimes it's string for different systems,
                # so when it is a string we need to change target_attrs back to string
                self.target_attrs = target_attrs if isinstance(target_attrs, list) else [target_attrs]
                if domain_attrs is not None:
                    self.domain_attrs = domain_attrs if isinstance(domain_attrs, list) else [domain_attrs]
            self.y_index = [np.where(np.array(file["columns"]) == target_attr)[0][0] for target_attr in
                            self.target_attrs]
            if self.domain_attrs is not None:
                self.z_index = [np.where(np.array(file["columns"]) == domain_attr)[0][0] for domain_attr in
                                self.domain_attrs]
            self.labels = []
            self.total = file[self.true_type]['label'].shape[0]
            if type == "train":
                self.start_point = 0
                self.end_point = self.total - self.trigger_data_num
            elif type == "trigger":
                assert self.trigger_data_num > 0
                self.start_point = self.total - self.trigger_data_num
                self.end_point = self.total
            else:
                self.start_point = 0
                self.end_point = self.total
            for i in tqdm(range(self.start_point, self.end_point)):
                self.labels.append(file[self.true_type]['label'][i])
            self.lens = len(self.labels)

    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        # Do not open the file in the __init__, this will disable the num-workers.
        with h5py.File(self.root, mode='r') as file:
            # This is designed for "train" and "trigger", they share file["train"] but different start_point.
            # For "val" and "test", self.start_point + index = indx
            image = torch.Tensor(file[self.true_type]['data'][self.start_point + index] / 255.).permute(2, 0, 1)
            if self.img_transform is not None:
                image = self.img_transform(image)
            return image, self.get_label(index)

    def get_label(self, index):
        label_y = 0
        for i, y in enumerate(self.y_index):
            label_y += (2 ** i) * (int(self.labels[index][y]))
        label_z = 0
        if self.domain_attrs is not None:
            for i, z in enumerate(self.z_index):
                label_z += (2 ** i) * (int(self.labels[index][z]))
            return label_y, label_z
        return label_y


class CelebABalance(Dataset):
    def __init__(self, root, split='train', transform=None, num=None, base_ratio=1, add_aug_ratio=0.1,
                 add_aug_mag=.1, target_attr="Smiling", add_aug="rotation", domain_attr="Blond_Hair") -> None:
        """
        :param root: the path to the hdf5 file.
        :param split: [train, val, test]
        :param transform: the data transformation you want to use.
        :param num: the number of the total data.
        :param base_ratio: the ratio for the domain_attr, 1 for balance.
        :param add_aug_ratio: the ratio of the additional augmented data. Set to 0 if you do not want any data.
        :param add_aug: [rotation, angle, crop] The type of the augmentation.
        :param add_aug_mag: the magnitude for the additional augmentation. For rotation, max angle (-angle, angle); for crop, the remaining size; for gaussian noise, the variance.
        :param target_attr: the target attribute you want to use. The target_attr is automatically balanced. The input must be in the list "attr_list" on the top of this file.
        :param domain_attr: the domain attribute you want to use. The domain_attr is controlled by the param base_ratio.
        """
        super().__init__()
        assert add_aug in ["gaussian", "rotation", "crop"]
        self.add_aug = add_aug
        self.root = root
        self.split = split
        self.transform = transform
        self.target_attr = target_attr
        self.domain_attr = domain_attr

        # In case of bug like: IndexError: index 0 is out of bounds for axis 0 with size 0
        # Comment out the two lines below
        self.target_attr = bytes(target_attr, 'utf-8')
        self.domain_attr = bytes(self.domain_attr, 'utf-8')

        self.add_aug_mag = add_aug_mag
        with h5py.File(self.root, mode='r') as file:
            self.y_index = np.where(np.array(file["columns"]) == self.target_attr)[0][0]
            self.a_index = np.where(np.array(file["columns"]) == self.domain_attr)[0][0]
            labels = file[split]["label"][:, self.a_index] * 2 + file[split]["label"][:, self.y_index]
        indexes = [np.where(labels == i)[0] for i in range(4)]
        base_zero_min_idx = np.argmin(np.array([(labels == i).sum() for i in range(2)]))
        base_one_min_idx = np.argmin(np.array([(labels == i).sum() for i in range(2, 4)])) + 2

        total_min = len(indexes[base_one_min_idx]) if len(indexes[base_zero_min_idx]) > len(
            indexes[base_one_min_idx]) * base_ratio else int(len(indexes[base_zero_min_idx]) / base_ratio)
        if num is not None:
            num = int(num / 2)
            assert num < (total_min * (base_ratio + 1)) * 2, "No Enough Data, Lower The Total Num"
            total_min = int(num // (base_ratio + 1))
        indexes[0] = indexes[0][:int(total_min * base_ratio)]
        indexes[1] = indexes[1][:int(total_min * base_ratio)]
        indexes[2] = indexes[2][:total_min]
        indexes[3] = indexes[3][:total_min]
        gaussian_sample = random.sample(range(total_min), int(add_aug_ratio * total_min))
        indexes.append(indexes[2][gaussian_sample])
        indexes.append(indexes[3][gaussian_sample])
        self.aug_cutpoint = sum([len(indexes[i]) for i in range(4)])
        self.indexes = np.concatenate(indexes)

        self.num_classes = 2

    def __len__(self):
        return self.indexes.shape[0]

    def __getitem__(self, index):
        with h5py.File(self.root, mode='r') as file:
            img = torch.Tensor(file[self.split]['data'][self.indexes[index]] / 255.).permute(2, 0, 1)
            if self.transform != None:
                img = self.transform(img)
            if index >= self.aug_cutpoint:
                if self.add_aug == "gaussian":
                    img += torch.randn_like(img) * np.sqrt(self.add_aug_mag)
                elif self.add_aug == "crop":
                    self.add_aug_mag = int(self.add_aug_mag)
                    add_aug = Compose([RandomCrop(self.add_aug_mag), Resize(224)])
                    img = add_aug(img)
                else:
                    self.add_aug_mag = int(self.add_aug_mag)
                    add_aug = RandomRotation(self.add_aug_mag)
                    img = add_aug(img)
            label = int(file[self.split]['label'][self.indexes[index]][self.y_index])
            label_z = int(file[self.split]['label'][self.indexes[index]][self.a_index])
        return img, (label, label_z)


class CelebAMultiBalance(Dataset):
    def __init__(self, root, split='train', transform=None, num=None, base_ratio=1, add_aug_ratio=0.1,
                 add_aug_mag=.1, target_attrs=None, add_aug="rotation", domain_attr="Blond_Hair") -> None:
        super().__init__()
        assert add_aug in ["gaussian", "rotation", "crop"]
        assert target_attrs is not None
        for attr in target_attrs:
            assert attr in attr_list
        self.add_aug = add_aug
        self.root = root
        self.split = split
        self.transform = transform
        self.target_attrs = [bytes(attr, 'utf-8') for attr in target_attrs]
        self.domain_attr = bytes(domain_attr, 'utf-8')
        self.add_aug_mag = add_aug_mag

        with h5py.File(self.root, mode='r') as file:
            columns = np.array(file["columns"])
            self.y_indices = [np.where(columns == attr)[0][0] for attr in self.target_attrs]
            self.a_index = np.where(columns == self.domain_attr)[0][0]
            labels = np.sum(
                [(2 ** i) * file[split]["label"][:, self.y_indices[i]] for i in range(len(self.target_attrs))], axis=0)
            num_labels = 2 ** len(self.target_attrs)
            labels += num_labels * file[split]["label"][:, self.a_index]

        indexes = [np.where(labels == i)[0] for i in range(num_labels * 2)]
        total_min = min([len(indexes[i]) for i in range(num_labels)])

        if num is not None:
            num = int(num / num_labels)
            assert num < total_min, "No Enough Data, Lower The Total Num"
            total_min = num

        for i in range(num_labels):
            indexes[i] = indexes[i][:total_min * base_ratio]
            indexes[i + num_labels] = indexes[i + num_labels][:total_min]

        gaussian_sample = random.sample(range(total_min), int(add_aug_ratio * total_min))
        indexes.append(np.concatenate([indexes[i + num_labels][gaussian_sample] for i in range(num_labels)]))
        self.aug_cutpoint = sum([len(indexes[i]) for i in range(num_labels * 2)])
        self.indexes = np.concatenate(indexes)

        self.num_classes = num_labels

    def __len__(self):
        return self.indexes.shape[0]

    def __getitem__(self, index):
        with h5py.File(self.root, mode='r') as file:
            img = torch.Tensor(file[self.split]['data'][self.indexes[index]] / 255.).permute(2, 0, 1)
            if self.transform != None:
                img = self.transform(img)
            if index >= self.aug_cutpoint:
                if self.add_aug == "gaussian":
                    img += torch.randn_like(img) * np.sqrt(self.add_aug_mag)
                elif self.add_aug == "crop":
                    self.add_aug_mag = int(self.add_aug_mag)
                    add_aug = Compose([RandomCrop(self.add_aug_mag), Resize(224)])
                    img = add_aug(img)

                else:
                    self.add_aug_mag = int(self.add_aug_mag)
                    add_aug = RandomRotation(self.add_aug_mag)
                    img = add_aug(img)

            label = int(np.sum([(2 ** i) * file[self.split]['label'][self.indexes[index]][self.y_indices[i]] for i in
                                range(len(self.target_attrs))]))
            label_z = int(file[self.split]['label'][self.indexes[index]][self.a_index])

        return img, (label, label_z)


if __name__ == '__main__':
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--num-workers", "--n", type=int, default=2)
    # parser.add_argument("--data-dir", "--d", type=str, default='../data/CelebA/')
    # args = parser.parse_args()
    # data_dir = args.data_dir
    # num_workers = args.num_workers
    # print("================= Test Fast CelebA Dataset =================")
    # data = CelebAFast(os.path.join(data_dir, 'celeba.hdf5'), ['Blond_Hair', 'Smiling'], domain_attrs=['Male', 'Arched_Eyebrows'], type="train")
    # loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=num_workers)
    # for (img, (label, domain)) in tqdm(loader):
    #     pass
    #
    # print("================= Test Fast CelebA Dataset =================")
    # data = CelebAFast(os.path.join(data_dir, 'celeba.hdf5'), ['Blond_Hair', 'Smiling'], domain_attrs=['Male'],
    #                   type="test")
    # loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=num_workers)
    # for (img, (label, domain)) in tqdm(loader):
    #     print(torch.unique(label))

    # print("================= Test CelebA Dataset =================")
    # data = CelebA(data_dir, ['Blond_Hair'], domain_attrs=['Male'], type="train")
    # loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=num_workers)
    # for (img, (label, domain)) in tqdm(loader):
    #     pass
    #
    # print("================= Test Spurious CelebA Dataset =================")
    # spurious_data = CelebSpu(data_dir)
    # spurious_loader = DataLoader(spurious_data, batch_size=256, shuffle=False, num_workers=num_workers)
    # for (img, (label, domain)) in tqdm(loader):
    #     pass

    D = CelebABalance(f"../data/celeba/celeba.hdf5", add_aug_ratio=0.5)
    print(len(D), D.aug_cutpoint)
    print(D.__getitem__(106654)[0].shape)
