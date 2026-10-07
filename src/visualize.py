import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

import plot_style
from config import *
from model import get_model, load_pretrained
from utils import get_dataloader

plot_style.apply_style()


def plot_confusion_matrix(cm: list, classes: list[str], save_path: Path):
    """ 画混淆矩阵热力图 """

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    plot_style.save(fig, save_path)
    plt.close(fig)
    print(f"混淆矩阵已保存: {save_path}")


def plot_training_history(history_path: Path, save_path: Path):
    """ 画训练曲线：左边loss曲线，右边accuracy曲线 """

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["train_loss"], color=plot_style.PALETTE[0], label="Train")
    axes[0].plot(epochs, history["val_loss"], color=plot_style.PALETTE[1], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    plot_style.add_panel_label(axes[0], "(a)")

    axes[1].plot(epochs, history["train_acc1"], color=plot_style.PALETTE[0], label="Train")
    axes[1].plot(epochs, history["val_acc1"], color=plot_style.PALETTE[1], label="Val Top-1")
    axes[1].plot(epochs, history["val_acc5"], color=plot_style.PALETTE[2], label="Val Top-5")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    plot_style.add_panel_label(axes[1], "(b)")

    fig.tight_layout()
    save_path = save_path / "training_history.png"
    plot_style.save(fig, save_path)
    plt.close(fig)
    print(f"训练曲线已保存: {save_path}")


def visualize_dataset_samples(split_json, save_path: Path, num_samples=16):
    """ 展示模型实际"看到"的训练数据(经过数据增强之后的图片) """

    train_loader, _ = get_dataloader(split_json)
    images, labels = next(iter(train_loader))
    images = images[:num_samples]

    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    images = torch.clamp(images * std + mean, 0, 1)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.axis("off")

    fig.tight_layout()
    save_path = save_path / "dataset_samples.png"
    plot_style.save(fig, save_path)
    print(f"数据集样本图已保存: {save_path}")


def visualize_feature_maps(checkpoint_path: Path, split_json, save_path: Path, model_type: str):
    
    if not checkpoint_path.exists():
        print(f"检查点不存在: {checkpoint_path}，将使用随机初始化权重展示")
        model = get_model(model_type)
    else:
        model, _ = load_pretrained(model_type, checkpoint_path)
        print(f"已加载模型: {checkpoint_path}")

    _, val_loader = get_dataloader(split_json)
    image, label = next(iter(val_loader))
    image = image[:1].to(DEVICE)

    with open(split_json, "r") as f:
        class_names = json.load(f)["class_names"]

    feature_maps = {}

    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()  # detach()：不保留梯度信息，节省内存
        return hook

    handles = []
    layers_to_vis = [
        ("conv1", model.conv1),
        ("layer1", model.layer1),
        ("layer2", model.layer2),
        ("layer4", model.layer4),
    ]
    for name, layer in layers_to_vis:
        handles.append(layer.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        model(image)  # 触发一次前向传播，过程中会自动调用上面注册的hook

    for h in handles:
        h.remove()

    fig, axes = plt.subplots(1, len(layers_to_vis), figsize=(18, 5))
    for idx, (name, fmap) in enumerate(feature_maps.items()):
        fmap = fmap[0][:64].cpu().numpy()
        n_channels = min(64, fmap.shape[0])
        grid_size = int(np.ceil(np.sqrt(n_channels)))

        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)

        grid = np.zeros((grid_size * fmap.shape[1], grid_size * fmap.shape[2]))
        for i in range(n_channels):
            row, col = i // grid_size, i % grid_size
            grid[row*fmap.shape[1]:(row+1)*fmap.shape[1],
                 col*fmap.shape[2]:(col+1)*fmap.shape[2]] = fmap[i]

        axes[idx].imshow(grid, cmap="viridis")
        axes[idx].set_title(f"{name}\n({fmap.shape[0]} channels)")
        axes[idx].axis("off")

    fig.suptitle(f"Feature Maps — True Label: {class_names[label[0]]}", fontsize=14)
    fig.tight_layout()
    save_path = save_path /  "feature_maps.png"
    plot_style.save(fig, save_path)
    print(f"特征图已保存: {save_path}")


def visualize_training_history(save_path: Path, history_path: Path):
    """命令行入口用的薄封装"""

    if not history_path.exists():
        print(f"训练历史不存在: {history_path}，请先运行训练: python train.py")
        return
    plot_training_history(history_path, save_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--model-type", type=str, default="resnet50",
            choices=["resnet50", "4_SE", "3SE_1incep", "resnext50"],
            help="可选模型类型"
        )
    parser.add_argument("--mode", choices=["dataset", "features", "history"], default="dataset")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split-json", type=str, default=SPLIT)
    args = parser.parse_args()

    # checkpoint 没显式传就按 model_type 拼默认值
    checkpoint_path = Path(args.checkpoint) if args.checkpoint \
        else CHECKPOINTS / args.model_type / "best.pth"

    figures_dir = RESULTS / args.model_type / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "dataset":
        visualize_dataset_samples(args.split_json, figures_dir)
    elif args.mode == "features":
        visualize_feature_maps(checkpoint_path, args.split_json,
                               figures_dir, args.model_type)
    elif args.mode == "history":
        history_path = RESULTS / args.model_type / "history.json"
        visualize_training_history(figures_dir, history_path)


if __name__ == "__main__":
    main()