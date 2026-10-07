import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

import plot_style
from config import *
from model import get_model
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


def plot_training_history(history_path, save_path):
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
    axes[1].plot(epochs, history["val_acc1"], color=plot_style.PALETTE[1], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 Accuracy (%)")
    axes[1].legend()
    plot_style.add_panel_label(axes[1], "(b)")

    fig.tight_layout()
    plot_style.save_figure(fig, save_path)
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