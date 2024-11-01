import torch
from os import path
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
from torch._prims_common import DeviceLikeType
from typing import Union, Dict, Any, Optional, Sequence, Callable, Iterable, Tuple
import numpy as np
from sklearn import metrics
from config import load_args
import datasets.imagenet_folder as imagenet_folder 
from models import resnet110
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import tqdm
loader_t = DataLoader[Union[torch.Tensor, torch.Tensor]]

class ClassificationMeter:
    def __init__(self, num_classes: int, record_logits: bool = False) -> None:
        self.num_classes = num_classes
        self.total_loss = 0.0
        self.labels = np.zeros((0,), dtype=np.int32)
        self.prediction = np.zeros((0,), dtype=np.int32)
        self.acc5_cnt = 0
        self.record_logits = record_logits
        if self.record_logits:
            self.logits = np.ndarray((0, num_classes))

    def record(self, y_true: torch.Tensor, logits: torch.Tensor) -> None:
        self.labels = np.concatenate([self.labels, y_true.cpu().numpy()])
        # Record logits
        if self.record_logits:
            logits_softmax = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            self.logits = np.concatenate([self.logits, logits_softmax])

        # Loss
        self.total_loss += float(
            torch.nn.functional.cross_entropy(logits, y_true, reduction="sum").item()
        )
        # Top-5 accuracy
        y_pred = logits.topk(5, largest=True).indices.to(torch.int)
        acc5_judge = (y_pred == y_true[:, None]).any(dim=-1)
        self.acc5_cnt += int(acc5_judge.sum().item())

        # Record the predictions
        self.prediction = np.concatenate([self.prediction, y_pred[:, 0].cpu().numpy()])

    @property
    def accuracy(self) -> float:
        return float(metrics.accuracy_score(self.labels, self.prediction))

    @property
    def balanced_accuracy(self) -> float:
        result = metrics.balanced_accuracy_score(
            self.labels, self.prediction, adjusted=True
        )
        return float(result)

    @property
    def f1_micro(self) -> float:
        result = metrics.f1_score(self.labels, self.prediction, average="micro")
        return float(result)

    @property
    def f1_macro(self) -> float:
        result = metrics.f1_score(self.labels, self.prediction, average="macro")
        return float(result)

    @property
    def accuracy5(self) -> float:
        return self.acc5_cnt / len(self.labels)

    @property
    def loss(self) -> float:
        return float(self.total_loss / len(self.labels))

@torch.no_grad()
def validate(
    model: Callable[[torch.Tensor], torch.Tensor],
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    desc: Optional[str] = None
) -> ClassificationMeter:
    if isinstance(model, torch.nn.Module):
        model.eval()
        device = next(model.parameters()).device
    else:
        device = model.device
    meter = ClassificationMeter(num_classes)

    for X, y in data_loader:
    
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
     
        # Calculate the loss
        logits: torch.Tensor = model(X)
        meter.record(y, logits)
    return meter

def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    disable_norm_decay: bool = True,
    disable_bias_decay: bool = True,
    disable_embedding_decay: bool = True,
):
    # See: https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    norm_classes = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    )

    params = {
        "other": [],
        "norm": [],
        "bias": [],
        "class_token": [],
        "position_embedding": [],
        "relative_position_bias_table": [],
    }

    params_weight_decay = {
        "bias": 0 if disable_bias_decay else weight_decay,
        "class_token": 0 if disable_embedding_decay else weight_decay,
        "position_embedding": 0 if disable_embedding_decay else weight_decay,
        "relative_position_bias_table": 0 if disable_embedding_decay else weight_decay,
    }

    def _add_params(module: torch.nn.Module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            for key in params_weight_decay.keys():
                target_name = (
                    f"{prefix}.{name}" if prefix != "" and "." in key else name
                )
                if key == target_name:
                    params[key].append(p)
                    break
            else:
                if isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    params_weight_decay["other"] = weight_decay
    params_weight_decay["norm"] = 0.0 if disable_norm_decay else weight_decay

    param_groups: List = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append(
                {"params": params[key], "weight_decay": params_weight_decay[key]}
            )
    return param_groups

class SubLearner:
   
    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        self.args = args
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.device = device
        self.all_devices = all_devices
        self.learning_rate: float = args["learning_rate"]
        self.buffer_size: int = args["buffer_size"]
        self.gamma: float = args["gamma"]
        self.base_epochs: int = args["base_epochs"]
        self.warmup_epochs: int = args["warmup_epochs"]
       
    def save_object(self, model, file_name: str) -> None:
        torch.save(model, path.join(self.args["saving_root"], file_name))
    def base_training(
        self,
        train_loader: loader_t,
        val_loader: loader_t,
        baseset_size: int,
    ) -> None:
        model = torch.nn.Sequential(
            self.backbone,
            torch.nn.Linear(self.backbone_output, baseset_size),
        ).to(self.device, non_blocking=True)
       

        if self.args["separate_decay"]:
            params = set_weight_decay(model, self.args["weight_decay"])
        else:
            params = model.parameters()
        optimizer = torch.optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.args["momentum"],
            weight_decay=self.args["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.base_epochs - self.warmup_epochs, eta_min=1e-6 # type: ignore
        )
        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=self.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, scheduler], [self.warmup_epochs]
            )

        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.args["label_smoothing"]
        ).to(self.device, non_blocking=True)

        best_acc = 0.0
        logging_file_path = path.join(self.args["saving_root"], "base_training.csv")
        logging_file = open(logging_file_path, "w", buffering=1)
        print(
            "epoch",
            "best_acc@1",
            "loss",
            "acc@1",
            "acc@5",
            "f1-micro",
            "training_loss",
            "training_acc@1",
            "training_acc@5",
            "training_f1-micro",
            "training_learning-rate",
            file=logging_file,
            sep=",",
        )

        for epoch in range(self.base_epochs + 1):
            if epoch != 0:
                print(
                    f"Base Training - Epoch {epoch}/{self.base_epochs}",
                    f"(Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']})",
                )
                model.train()
                for X, y in train_loader:
                   
                    X: torch.Tensor = X.to(self.device, non_blocking=True)
                    y: torch.Tensor = y.to(self.device, non_blocking=True)
                   
                  
                  
                    assert y.max() < baseset_size

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(X)
                    loss: torch.Tensor = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            # Validation on training set
            model.eval()
            train_meter = validate(
                model, train_loader, baseset_size, desc="Training (Validation)"
            )
            print(
                f"loss: {train_meter.loss:.4f}",
                f"acc@1: {train_meter.accuracy * 100:.3f}%",
                f"acc@5: {train_meter.accuracy5 * 100:.3f}%",
                f"f1-micro: {train_meter.f1_micro * 100:.3f}%",
                sep="    ",
            )

            val_meter = validate(model, val_loader, baseset_size, desc="Testing")
            if val_meter.accuracy > best_acc:
                best_acc = val_meter.accuracy
                if epoch != 0:
                    self.save_object(
                        (self.backbone, X.shape[1], self.backbone_output),
                        "backbone.pth",
                    )

            # Validation on testing set
            print(
                f"loss: {val_meter.loss:.4f}",
                f"acc@1: {val_meter.accuracy * 100:.3f}%",
                f"acc@5: {val_meter.accuracy5 * 100:.3f}%",
                f"f1-micro: {val_meter.f1_micro * 100:.3f}%",
                f"best_acc@1: {best_acc * 100:.3f}%",
                sep="    ",
            )
            print(
                epoch,
                best_acc,
                val_meter.loss,
                val_meter.accuracy,
                val_meter.accuracy5,
                val_meter.f1_micro,
                train_meter.loss,
                train_meter.accuracy,
                train_meter.accuracy5,
                train_meter.f1_micro,
                optimizer.state_dict()["param_groups"][0]["lr"],
                file=logging_file,
                sep=",",
            )
        logging_file.close()
        self.backbone.eval()
        

   
args=load_args()
args["base_epochs"]=300
if args["cpu_only"] or not torch.cuda.is_available():
        main_device = torch.device("cpu")
        all_gpus = None
else:
        main_device = torch.device("cuda:1")
        all_gpus = None

transform = transforms.ToTensor()
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # normalize = transforms.Normalize((0.4810, 0.4574, 0.4078), (0.2146, 0.2104, 0.2138))

train_transform = transforms.Compose(
    [   transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        normalize,
    ]
)

test_transform = transforms.Compose([ transforms.ToTensor(),normalize])








whole_class=100
backbone=resnet110(num_classes=int(whole_class/2))
feature_size = backbone.fc.in_features
backbone.fc = torch.nn.Identity()  # type: ignore
train_set=imagenet_folder.ImageNetDS(
        root=args["data_root"], img_size=32, train=True, transform=train_transform
    )
test_set=imagenet_folder.ImageNetDS(
        root=args["data_root"], img_size=32, train=False, transform=test_transform
    )

class_indices = torch.tensor([i for i in range(50)])

subset_indices = []
for c in class_indices:
    class_mask = (torch.Tensor(train_set.targets).data == c.data)
    class_mask_tensor = torch.tensor(class_mask.clone().detach(), dtype=torch.int)  # 将布尔类型的class_mask转换为Tensor
    class_samples = torch.nonzero(class_mask_tensor).view(-1)
    subset_indices.extend(class_samples.tolist())
train_set_i = Subset(train_set, subset_indices)

subset_indices = []
for c in class_indices:
    class_mask = (torch.Tensor(test_set.targets).data == c.data)
    class_mask_tensor = torch.tensor(class_mask.clone().detach(), dtype=torch.int)  # 将布尔类型的class_mask转换为Tensor
    class_samples = torch.nonzero(class_mask_tensor).view(-1)
    subset_indices.extend(class_samples.tolist())
test_set_i = Subset(test_set, subset_indices)



train_loader = DataLoader(dataset=train_set_i, batch_size=128 , shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_set_i , batch_size=128 , shuffle=False, num_workers=4)
sum=0

learner = SubLearner(
        args, backbone, feature_size, main_device, all_devices=all_gpus
    )

learner.base_training(train_loader,test_loader,int(whole_class/2))