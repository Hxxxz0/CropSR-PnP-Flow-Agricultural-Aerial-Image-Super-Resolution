import torch
import torchvision
import torchvision.transforms as v2
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import pandas as pd
import os
import warnings
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pickle
import logging


class DataLoaders:
    def __init__(self, dataset_name, batch_size_train, batch_size_test):
        self.dataset_name = dataset_name
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

    def load_data(self):

        if self.dataset_name == 'celeba':
            transform = v2.Compose([
                v2.CenterCrop(178),
                v2.Resize((128, 128)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            # Paths
            img_dir = './data/celeba/img_align_celeba/'
            partition_csv = './data/celeba/list_eval_partition.csv'

            # Datasets
            train_dataset = CelebADataset(
                img_dir, partition_csv, partition=0, transform=transform)
            val_dataset = CelebADataset(
                img_dir, partition_csv, partition=1, transform=transform)
            test_dataset = CelebADataset(
                img_dir, partition_csv, partition=2, transform=transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        elif self.dataset_name == 'celebahq':

            transform = v2.Compose([
                v2.Resize(256),
                v2.ToTensor(),         # Convert images to PyTorch tensor
            ])

            test_dir = './data/celebahq/test/'
            test_dataset = CelebAHQDataset(
                test_dir, batchsize=self.batch_size_test, transform=transform)
            train_loader = None
            val_loader = None
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        elif self.dataset_name == 'afhq_cat':
            # transform should include a linear transform 2x - 1
            transform = v2.Compose([
                v2.Resize((256, 256)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            # transform = False
            img_dir_test = './data/afhq_cat/test/cat/'
            img_dir_val = './data/afhq_cat/val/cat/'
            img_dir_train = './data/afhq_cat/train/cat/'
            test_dataset = AFHQDataset(
                img_dir_test, batchsize=self.batch_size_test, transform=transform)
            val_dataset = AFHQDataset(
                img_dir_val, batchsize=self.batch_size_test, transform=transform)
            train_dataset = AFHQDataset(
                img_dir_train, batchsize=self.batch_size_test, transform=transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate, drop_last=True)

        elif self.dataset_name == 'cropsr':
            # HR图像变换：将原始512x512保持为512x512
            hr_transform = v2.Compose([
                v2.Resize((512, 512)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # LR图像变换：将原始256x256插值到512x512以匹配UNet输入
            lr_transform = v2.Compose([
                v2.Resize((512, 512), interpolation=v2.InterpolationMode.BICUBIC),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            hr_dir = '../CropSR (for Crop Field Aerial Image Super Resolution)/cropSR_ortho/HR/'
            lr_dir = '../CropSR (for Crop Field Aerial Image Super Resolution)/cropSR_ortho/LR/'
            
            # 创建数据集实例
            full_dataset = CropSRDataset(hr_dir, lr_dir, hr_transform=hr_transform, lr_transform=lr_transform)
            
            # 划分训练集、验证集、测试集 (8:1:1)
            total_size = len(full_dataset)
            train_size = int(0.8 * total_size)
            val_size = int(0.1 * total_size)
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate, drop_last=True)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        else:
            raise ValueError("The dataset your entered does not exist")

        data_loaders = {'train': train_loader,
                        'test': test_loader, 'val': val_loader}

        return data_loaders


class CelebADataset(Dataset):
    def __init__(self, img_dir, partition_csv, partition, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.partition = partition

        # Load the partition file correctly
        partition_df = pd.read_csv(
            partition_csv, header=0, names=[
                'image', 'partition'], skiprows=1)
        self.img_names = partition_df[partition_df['partition']
                                      == partition]['image'].values

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


class CelebAHQDataset(Dataset):
    """CelebA HQ dataset."""

    def __init__(self, data_dir, batchsize, transform=None):
        self.files = os.listdir(data_dir)
        self.root_dir = data_dir
        self.num_imgs = len(os.listdir(self.root_dir))
        self.transform = transform
        self.batchsize = batchsize

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            image = 2 * image - 1
        image = image.float()

        return image, 0


class AFHQDataset(Dataset):
    """AFHQ Cat dataset."""

    def __init__(self, img_dir, batchsize, category='cat', transform=None):
        self.files = os.listdir(img_dir)
        self.num_imgs = len(self.files)
        self.batchsize = batchsize
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


class CropSRDataset(Dataset):
    """CropSR Agricultural Aerial Image dataset for Super Resolution."""

    def __init__(self, hr_dir, lr_dir, hr_transform=None, lr_transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        
        # 获取所有HR图像文件名
        self.hr_files = [f for f in os.listdir(hr_dir) if f.endswith('.jpg')]
        self.hr_files.sort()  # 确保顺序一致
        
        # 验证LR图像存在
        self.valid_files = []
        for hr_file in self.hr_files:
            lr_path = os.path.join(lr_dir, hr_file)
            if os.path.exists(lr_path):
                self.valid_files.append(hr_file)
        
        print(f"CropSR Dataset: Found {len(self.valid_files)} valid HR-LR pairs")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        img_name = self.valid_files[idx]
        hr_path = os.path.join(self.hr_dir, img_name)
        lr_path = os.path.join(self.lr_dir, img_name)

        if not os.path.exists(hr_path) or not os.path.exists(lr_path):
            warnings.warn(f"File not found: {hr_path} or {lr_path}. Skipping.")
            return None, None

        # 加载HR和LR图像
        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = Image.open(lr_path).convert('RGB')

        # 应用对应的变换
        if self.hr_transform:
            hr_image = self.hr_transform(hr_image)
        if self.lr_transform:
            lr_image = self.lr_transform(lr_image)

        # 对于超分辨率任务，我们需要返回HR图像作为目标
        # 注意：这里返回HR图像作为训练目标，LR图像将在超分辨率推理时使用
        return hr_image, 0


def custom_collate(batch):
    # Filter out None values

    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data._utils.collate.default_collate(batch)


logging.basicConfig(level=logging.INFO)
