# -*- coding: utf-8 -*-
from typing import Tuple
import torch
from .DatasetWrapper import DatasetWrapper
from torchvision.datasets import ImageNet
from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2 as transforms
from os import path
import data.imagenet_folder as imagenet_folder 
class CustomDataset(Dataset):

    def __init__(self, subset):
        self.subset = subset
        self.dataset = subset.dataset  # 原始数据集
        self.tensors = self.dataset.tensors

    def __getitem__(self, index):
        return self.dataset[self.subset.indices[index]]

    def __len__(self):
        return len(self.subset)

    @property
    def targets(self):
        # 使用列表推导式来获取子集的标签
        return [self.dataset.targets[i] for i in self.subset.indices]
    
class ImageNet_(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 100
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    basic_transform = transforms.Compose(
        [   transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    normalize,
    ])
    test_transform = transforms.Compose([ transforms.ToTensor(),normalize])

    def __init__(
        self,
        root: str,
        train: bool,
        base_ratio: float,
        num_phases: int,
        augment: bool = False,
        inplace_repeat: int = 1,
        shuffle_seed: int | None = None,
        divide_number:int=5
    ) -> None:

        root = path.expanduser(root)
        class_indices = torch.tensor([i for i in range(85)])
        subset_indices = []
        if train:
            self.dataset = imagenet_folder.ImageNetDS(root=root, img_size=32,train=True, transform=None)
            for c in class_indices:
                class_mask = (torch.Tensor(self.dataset.targets).data == c.data)
                class_mask_tensor = torch.tensor(class_mask.clone().detach(), dtype=torch.int)  # 将布尔类型的class_mask转换为Tensor
                class_samples = torch.nonzero(class_mask_tensor).view(-1)
               
                subset_indices.extend(class_samples.tolist())
          
            self.dataset = CustomDataset(Subset(self.dataset, subset_indices))
        else:
            self.dataset = imagenet_folder.ImageNetDS(root=root, img_size=32,train=False, transform=None)
            for c in class_indices:
                class_mask = (torch.Tensor( self.dataset .targets).data == c.data)
                class_mask_tensor = torch.tensor(class_mask.clone().detach(), dtype=torch.int)  # 将布尔类型的class_mask转换为Tensor
                class_samples = torch.nonzero(class_mask_tensor).view(-1)
                subset_indices.extend(class_samples.tolist())
            self.dataset=  CustomDataset(Subset(self.dataset, subset_indices))
        super().__init__(
            self.dataset.targets,
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
            divide_number=divide_number
        )
