import torch
import random
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from config import *


class IN100Dataset(Dataset):
    """构建 ImageNet-100 数据集"""

    def __init__(self, split_path: Path, split_name: str, transform: T.Compose):
        super().__init__()
        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = data[split_name] # (path, number)
        self.classes = data["classes"]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
    
def get_dataloader(split_path: Path):
    """构建训练和验证用的 DataLoader"""

    transform_train = T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.RandAugment(NUM_OPS),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=RANDOM_ERASING_P)
    ])
    transform_val = T.Compose([
        T.Resize(256),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])

    train_set = IN100Dataset(split_path, "train", transform_train)
    val_set = IN100Dataset(split_path, "val", transform_val)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # 防影响 BatchNorm 统计精度
        prefetch_factor=PREFECTCH_FACTOR,
        persistent_workers=True,  # 只创建一次 worker 进程
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFECTCH_FACTOR,
        persistent_workers=True
    )

    return train_loader, val_loader

def worker_init_fn(worker_id):
    """设置子进程的随机种子"""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)

def set_seed(seed=SEED):
    """使实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 提高可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter():
    """滑动平均指标计算器, 用于追踪 loss 和 accuracy"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = self.count = self.avg = 0.0

    def update(self, val, n=1):
        """更新滑动平均指标"""
        if isinstance(val, torch.Tensor):
            val = val.item() # 处理张量

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        topk: tuple[int, ...] = (1,)
    ) -> list[torch.Tensor]:
    """
    计算 top-k 准确率
    
    Args:
        outputs:  模型输出的 logits，(batch_size, num_classes)
        targets:  真实标签，(batch_size,)
        topk:     要计算哪些 k 值的准确率
    
    Return:
        一个列表，每个元素是对应 k 值的准确率（百分比形式）
    """

    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = outputs.topk(maxk, dim=1)
    pred = pred.t() # (maxk, batch_size)
    correct = pred == targets.view(1, -1).expand_as(pred)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        res.append(correct_k* 100.0 / batch_size)
    return res
