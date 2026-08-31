import random
import json
from pathlib import Path
from config import SEED, TRAIN_VAL_SPLIT_RATIO, TRAIN, LABELS, DATA_ROOT

def train_val_split(ratio: float = TRAIN_VAL_SPLIT_RATIO, seed: int = SEED):
    """
    划分训练集和验证集
    
    args:
        ratio:       训练集占比，比如 0.95 表示 95% 训练 / 5% 验证
        seed:        随机种子
    """

    random.seed(seed)
    train_samples, val_samples = [], []

    with open(LABELS, "r", encoding="utf-8") as f:
        labels = json.load(f)
    
    # 用sorted保证序列对应
    classes = sorted(i.name for i in TRAIN.iterdir() if i.is_dir())
    class_names = [labels[i] for i in classes]

    for i, cls in enumerate(classes):
        class_dir = TRAIN / cls
        files = [f for f in class_dir.iterdir() if f.is_file()]
        random.shuffle(files)
        indexes = int(len(files) * ratio)

        train_samples += [(str(f), i) for f in files[: indexes]]
        val_samples += [(str(f), i) for f in files[indexes :]]
    
    # 存成json形式
    with open(DATA_ROOT / "split.json", "w", encoding="utf-8") as f:
        json.dump({
            "train": train_samples,
            "val": val_samples,
            "classes": classes,
            "class_names": class_names
        }, f)
    
    print(f"已按比例{ratio}划分好train和val为一文件 {DATA_ROOT / 'split.json'}")
    print(f"train: {len(train_samples)} 张, val: {len(val_samples)} 张")
    print()


if __name__ == "__main__":
    train_val_split()