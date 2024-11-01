import math
import random
import functools
from datasets import Features, load_dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import numpy as np
import logging
import torch.distributed as dist
from typing import Any, Dict, List, Tuple
import os
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data import DataLoader
from typing import Optional, Union, Callable
from os import path
from tqdm import tqdm
from datasets.ImageNet import CustomDataset
activation_t = Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RandomBuffer(torch.nn.Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=torch.double,
        activation: Optional[activation_t] = torch.relu_,
    ) -> None:
        super(torch.nn.Linear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.activation: activation_t = (torch.nn.Identity()
                                         if activation is None else activation)

        W = torch.empty((out_features, in_features), **factory_kwargs)
        b = torch.empty(out_features, **factory_kwargs) if bias else None

        # Using buffer instead of parameter
        self.register_buffer("weight", W)
        self.register_buffer("bias", b)

        # Random Initialization
        self.reset_parameters()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
        return self.activation(super().forward(X))


class DataPartitioner(object):
    """Partitions a dataset into different chuncks."""

    def __init__(self, dataset, partition_sizes, partition_alphas, seed,n_c):

        self.seed = seed
        self.partition_sizes = partition_sizes
        self.partition_alphas = partition_alphas
        self.data = dataset
        self.partitions = []
        self.data_size = len(dataset)
        indices = np.array([x for x in range(0, self.data_size)])
        num_classes = n_c  #len(np.unique(self.data.targets))
        n_workers = len(self.partition_sizes)
        indices2targets = np.array([
            (idx, target) for idx, target in enumerate(self.data.targets)
            if idx in indices
        ])
        non_iid_alphas = ([float(x) for x in self.partition_alphas.split(":")]
                          if type(self.partition_alphas) is not float else
                          [self.partition_alphas])
        list_of_indices = []
        assert n_workers % len(non_iid_alphas) == 0
        assert self.data_size % len(non_iid_alphas) == 0
        num_sub_indices = int(self.data_size / len(non_iid_alphas))

        for idx, non_iid_alpha in enumerate(non_iid_alphas):
            _list_of_indices = build_non_iid_by_dirichlet(
                indices2targets=indices2targets[int(idx * num_sub_indices):int(
                    (idx + 1) * num_sub_indices)],
                non_iid_alpha=non_iid_alpha,
                num_classes=num_classes,
                num_indices=num_sub_indices,
                n_workers=int(n_workers / len(non_iid_alphas)),
                seed=self.seed)
            list_of_indices += _list_of_indices
        indices = functools.reduce(lambda a, b: a + b, list_of_indices)
        from_index = 0
        for partition_size in self.partition_sizes:
            to_index = from_index + int(partition_size * self.data_size)
            self.partitions.append(indices[from_index:to_index])
            from_index = to_index
        label_hist, hist = record_class_distribution(self.partitions,
                                                     self.data.targets,num_classes )

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind])


def build_non_iid_by_dirichlet(indices2targets, non_iid_alpha, num_classes,
                               num_indices, n_workers, seed):
    n_auxi_workers = 10
    state = np.random.RandomState(seed)
    state.shuffle(indices2targets)
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers if idx < num_splits - 1 else n_workers -
        n_auxi_workers * (num_splits - 1) for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[from_index:(num_indices if idx == num_splits -
                                        1 else to_index)])
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size == 0:
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers))
                    # balance
                    proportions = np.array([
                        p * (len(idx_j) < _targets_size / _n_workers)
                        for p, idx_j in zip(proportions, _idx_batch)
                    ])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) *
                                   len(idx_class)).astype(int)[:-1]
                    _idx_batch = [
                        idx_j + idx.tolist() for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions))
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


def record_class_distribution(partitions, targets,  num_cls=100,record=False, ):
    targets_of_partitions_w_labels = {}
    targets_of_partitions = []

    targets_np = np.array(targets)
    # compute unique values here
  
    for idx, partition in enumerate(partitions):
        unique_elements, counts_elements = np.unique(targets_np[partition],
                                                     return_counts=True)
        targets_of_partitions_w_labels[idx] = list(
            zip(unique_elements, counts_elements))
        temp = np.zeros(num_cls, dtype=int)
        temp[unique_elements] = counts_elements
        targets_of_partitions.append(list(temp))
    if record:
        print(
            f"the histogram of the targets in the partitions: {targets_of_partitions_w_labels.items()}\n"
        )
    return targets_of_partitions_w_labels, targets_of_partitions





def calculate_class_accuracies(model, test_loader, num_classes):
    """
    计算测试集上每个类别的准确率向量。

    参数:
    - model: 训练好的 PyTorch 模型。
    - test_loader: 测试数据的 DataLoader。
    - num_classes: 数据集中的类别总数。

    返回:
    - accuracies: 每个类别的准确率向量。
    """
    model.eval()  # 将模型设置为评估模式
    correct_counts = torch.zeros(num_classes, dtype=torch.int64)
    total_counts = torch.zeros(num_classes, dtype=torch.int64)

    with torch.no_grad():  # 不计算梯度
        for images, labels in test_loader:
            images, labels = images.to(next(
                model.parameters()).device), labels.to(
                    next(model.parameters()).device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # 统计每个类别预测正确的数量
            for label in labels:
                total_counts[label] += 1
                if predicted[labels == label].sum().item() > 0:
                    correct_counts[label] += predicted[labels ==
                                                       label].sum().item()

    # 计算准确率
    accuracies = correct_counts / total_counts
    return accuracies


class Partition(object):
    """Dataset-like object, but only access a subset of it."""

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices
        self.replaced_targets = None
        self.tensors = self.data.tensors

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        if self.replaced_targets is None:
            return self.data[data_idx]
        else:
            return (self.data[data_idx][0], self.replaced_targets[index])

    def update_replaced_targets(self, replaced_targets):
        self.replaced_targets = replaced_targets

        # evaluate the the difference between original labels and the simulated labels.
        count = 0
        for index in range(len(replaced_targets)):
            data_idx = self.indices[index]

            if self.replaced_targets[index] == self.data[data_idx][1]:
                count += 1
        return count / len(replaced_targets)

    def clean_replaced_targets(self):
        self.replaced_targets = None


def make_dataloader(dataset: Dataset,
                    shuffle: bool = False,
                    batch_size: int = 256,
                    num_workers: int = 8) -> DataLoader:
    config = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": DEVICE.type == "cuda",
        "pin_memory_device": str(DEVICE) if DEVICE.type == "cuda" else "",
    }

    return DataLoader(dataset, **config)


@torch.no_grad()
def cache_features(
    backbone: torch.nn.Module, dataloader: DataLoader[Tuple[torch.Tensor,
                                                            torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    X_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    for X, y in tqdm(dataloader, "Caching"):
        X: torch.Tensor = backbone(X.to(DEVICE))
        y: torch.Tensor = y.to(torch.int16, non_blocking=True)
        X_all.append(X.to("cpu", non_blocking=True))
        y_all.append(y)
    return torch.cat(X_all), torch.cat(y_all)


def check_cache_features(root: str) -> bool:
    files_list = ["X_train.pt", "y_train.pt", "X_test.pt", "y_test.pt"]
    for file in files_list:
        if not path.isfile(path.join(root, file)):
            return False
    return True


def prepare_dataloader(a, ratio, K, partition_sizes, partition_alphas, seed,data_name):
    if data_name=='C':
        args = {
            "dataset": "CIFAR-100",
            "data_root": "~/.dataset", #use own data root
            "IL_batch_size": 4096,
            "batch_size": 256,
            "num_workers": 8,
            "cache_path": "./backbones/resnet32_CIFAR-100_0.5_None",
            "backbone_path": "./backbones/resnet32_CIFAR-100_0.5_None",
        }
        FCL=100
    else:
        args = {
            "dataset": "ImageNet",
            "data_root": "~/.dataset", #use own data root
            "IL_batch_size": 4096,
            "batch_size": 256,
            "num_workers": 8,
            "cache_path": "./backbones/resnet32_downsampled_imagenet_0.5_None",
            "backbone_path": "./backbones/resnet32_downsampled_imagenet_0.5_None",
        }
        FCL=85
    backbone_path = path.join(args["backbone_path"], "backbone.pth")
    backbone, _, feature_size = torch.load(backbone_path, map_location=DEVICE)
    backbone = backbone.to(DEVICE, non_blocking=True).eval()

    if not check_cache_features(args["cache_path"]):
        dataset_train_cache = load_dataset(args["dataset"],
                                           args["data_root"],
                                           True,
                                           FCL,
                                           0,
                                           augment=False,
                                           divide_number=a)
        dataset_test_cache = load_dataset(args["dataset"],
                                          args["data_root"],
                                          False,
                                          FCL,
                                          0,
                                          augment=False,
                                          divide_number=a)
        train_loader = make_dataloader(
            dataset_train_cache.subset_at_phase(0),
            False,
            args["batch_size"],
            args["num_workers"],
        )
        test_loader = make_dataloader(
            dataset_test_cache.subset_at_phase(0),
            False,
            args["batch_size"],
            args["num_workers"],
        )
        X_train, y_train = cache_features(backbone, train_loader)
        X_test, y_test = cache_features(backbone, test_loader)
        torch.save(X_train, path.join(args["cache_path"], "X_train.pt"))
        torch.save(y_train, path.join(args["cache_path"], "y_train.pt"))
        torch.save(X_test, path.join(args["cache_path"], "X_test.pt"))
        torch.save(y_test, path.join(args["cache_path"], "y_test.pt"))

    dataset_train = Features(args["cache_path"],
                             train=True,
                             base_ratio=50,
                             num_phases=0,
                             augment=False,
                             divide_number=len(ratio))
    
    realign_set = CustomDataset(dataset_train.subset_at_phase(0))
    re_align_loader = make_dataloader(realign_set, True, len(realign_set),
                                      args["num_workers"])

    dataset_test = Features(args["cache_path"],
                            train=False,
                            base_ratio=FCL,
                            num_phases=0,
                            augment=False,
                            divide_number=len(ratio)).subset_at_phase(0)

    server_loader = make_dataloader(dataset_test, True, len(dataset_test),
                                    args["num_workers"])

    indices_of_dos = dataset_train.get_divideed_indices(ratio, seed)

    do_datasets = []

    for do_index in range(a):
        do_datasets.append(
            CustomDataset(Subset(dataset_train, indices_of_dos[do_index])))

    DataPartitioners_for_dos = []

    divided_datasets = [[] for i in range(a)]
    divided_data_loaders = [[] for i in range(a)]
    for do_index in range(a):
        DataPartitioners_for_dos.append(
            DataPartitioner(do_datasets[do_index], partition_sizes[do_index],
                            partition_alphas[do_index], seed,FCL))
        for task_index in range(K):
            divided_datasets[do_index].append(
                DataPartitioners_for_dos[do_index].use(task_index))

    for do_index in range(a):
        for task_index in range(K):
            divided_data_loaders[do_index].append(
                make_dataloader(divided_datasets[do_index][task_index], True,
                                len(divided_datasets[do_index][task_index]),
                                args["num_workers"]))

    return re_align_loader, server_loader, divided_data_loaders, feature_size


def AFL(dataset, c, buffer, rg, fs,ratio):
    for X, Z in dataset:
        X = buffer(X)
        Z_scalar = Z.int()

        one_hot_matrix = torch.eye(c)
        Z = one_hot_matrix[Z_scalar].double()
        C = X.T @ X + rg * torch.eye(fs).double()
        R = torch.inverse(C)
        W = R @ X.T @ Z
    return (W, C, R,1,ratio)


def aggregate(history_model, to_sell_model):
    W = [history_model[0], to_sell_model[0]]
    C = [history_model[1], to_sell_model[1]]
    R = [history_model[2], to_sell_model[2]]
    Wt = (torch.eye(R[0].shape[0]).double() - R[0] @ C[1] +
          R[0] @ C[1] @ torch.inverse(C[0] + C[1]) @ C[1]) @ W[0] + (
              torch.eye(R[0].shape[0]).double() - R[1] @ C[0] +
              R[1] @ C[0] @ torch.inverse(C[0] + C[1]) @ C[0]) @ W[1]
    Ct = C[0] + C[1]
    Rt = torch.pinverse(Ct)
    return (Wt, Ct, Rt,history_model[3]+to_sell_model[3])


def RI(W, C, nc, rg, fs):
    R_origin = torch.pinverse(C +nc * rg * torch.eye(fs).double())
    Wt = W + (nc * rg * R_origin) @ W
    return Wt


