import argparse
import json
import torch
from datetime import datetime
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from PIL import Image
from config import *
from model import load_pretrained
from utils import set_seed, build_transforms
from torch.utils.data import DataLoader, Dataset
from visualize import plot_confusion_matrix


class ListDataset(Dataset):
    """从 (path, label) 样本列表构建的数据集（测试集用）。

    注意必须定义在模块顶层：Windows 上 DataLoader 开 num_workers>0 时
    用 spawn 方式创建子进程，dataset 对象要整个 pickle 后传过去，
    而函数内部定义的局部类无法被 pickle。
    """

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def load_test_loader(archive_dir, split_json):
    with open(split_json, "r") as f:
        split_data = json.load(f)
    classes = split_data["classes"]
    class_to_label = {c: i for i, c in enumerate(classes)}

    test_samples = []
    test_dir = archive_dir / "test"
    for cls_dir in sorted(test_dir.iterdir()):
        for img_path in sorted(cls_dir.iterdir()):
            test_samples.append((str(img_path), class_to_label[cls_dir.name]))

    test_set = ListDataset(test_samples, build_transforms(train=False))
    return DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS), classes


def evaluate_model(model, test_loader, device, classes):
    
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    correct = total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = 100.0 * correct / total
    print(f"\n测试集总体准确率: {acc:.2f}%")
    print("\n===== 分类报告 =====")
    print(classification_report(all_targets, all_preds, target_names=classes, digits=4))

    cm = confusion_matrix(all_targets, all_preds)
    return np.array(all_targets), np.array(all_preds), np.array(all_probs), cm


def compute_metrics(targets: np.ndarray, preds: np.ndarray,
                    probs: np.ndarray, cm: np.ndarray, classes: list[str]):
    n = len(targets)
    top1 = float((preds == targets).mean() * 100)

    top5_idx = np.argsort(-probs, axis=1)[:, :5]
    top5 = float((top5_idx == targets[:, None]).any(axis=1).mean() * 100)

    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    p = top1 / 100
    stderr = float(np.sqrt(p * (1 - p) / n) * 100)  # top-1 的标准误（百分点）

    return {
        "num_samples": int(n),
        "top1": top1,
        "top5": top5,
        "top1_stderr": stderr,
        "per_class_acc": {c: float(a * 100) for c, a in zip(classes, per_class_acc)},
    }


def save_results(result_dir: Path, metrics, extra, targets, preds,
                 probs: np.ndarray, cm, paths: list[str]):
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({**extra, **metrics}, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        result_dir / "test_outputs.npz",
        targets=targets,
        preds=preds,
        probs=probs.astype(np.float32),
        cm=cm,
        paths=np.array(paths),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type", type=str, default="resnet50",
        choices=["resnet50", "4_SE", "3SE_1incep", "resnext50"],
        help="可选模型类型"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split-json", type=str, default=SPLIT)
    args = parser.parse_args()

    set_seed()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint \
        else CHECKPOINTS / args.model_type / "best.pth"

    result_dir = RESULTS / args.model_type
    figures_dir: Path = result_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    test_loader, classes = load_test_loader(DATA_ROOT, args.split_json)

    if not checkpoint_path.exists():
        print(f"错误: 检查点不存在 {checkpoint_path}")
        print("请先运行: python train.py")
        return

    model, _ = load_pretrained(args.model_type, checkpoint_path)
    print(f"已加载模型: {checkpoint_path}")

    targets, preds, probs, cm = evaluate_model(model, test_loader, DEVICE, classes)

    metrics = compute_metrics(targets, preds, probs, cm, classes)
    print(f"\nTop-1: {metrics['top1']:.2f}% (标准误 ±{metrics['top1_stderr']:.2f})"
          f"  Top-5: {metrics['top5']:.2f}%")

    paths = [p for p, _ in test_loader.dataset.samples]
    extra = {
        "model_type": args.model_type,
        "checkpoint": str(checkpoint_path),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_results(result_dir, metrics, extra, targets, preds, probs, cm, paths)

    plot_confusion_matrix(cm, classes, figures_dir / "confusion_matrix.png")
    print("\n评估完成!")


if __name__ == "__main__":
    main()