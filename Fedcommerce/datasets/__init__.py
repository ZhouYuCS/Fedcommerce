# -*- coding: utf-8 -*-

from .DatasetWrapper import DatasetWrapper


from .CIFAR import CIFAR100_ as CIFAR100
from .ImageNet import ImageNet_ as ImageNet
from typing import Union
from .Features import Features


__all__ = [
    "load_dataset",
    "dataset_list",
 

    "CIFAR100",
    "ImageNet",
    "DatasetWrapper",
    "Features",
]

dataset_list = {
   

    "CIFAR-100": CIFAR100,
    "ImageNet": ImageNet,
}


def load_dataset(
    name: str,
    root: str,
    train: bool,
    base_ratio: float,
    num_phases: int,
    augment: bool = False,
    inplace_repeat: int = 1,
    shuffle_seed: int | None = None,
    divide_number:int=5,
    *args,
    **kwargs
) -> Union[ CIFAR100, ImageNet]:
    return dataset_list[name](
        root=root,
        train=train,
        base_ratio=base_ratio,
        num_phases=num_phases,
        augment=augment,
        inplace_repeat=inplace_repeat,
        shuffle_seed=shuffle_seed,
        divide_number=divide_number,
        *args,
        **kwargs
    )
