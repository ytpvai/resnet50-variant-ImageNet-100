import argparse
import logging
import torch
import json
import math
from pathlib import Path
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

from config import *
from model import  get_model
from utils import (get_dataloader, AverageMeter, set_seed, accuracy)


def setup_logging(logs_dir: Path):
    """配置 logging 模块"""

    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "train.log"),
            logging.StreamHandler()
        ]
    )


def get_lr(epoch: int, batch_idx: int, len_epoch: int) -> float:
    """
    线性 warmup 和余弦退火

    Args:
        epoch:       当前第几个 epoch
        batch_idx:   此 epoch 内第几个 step
        len_epoch:   一个 epoch 的 batch 数

    Return:
        当前 batch 的学习率
    """

    warmup_steps = WARMUP_EPOCHS * len_epoch
    current_step = len_epoch * epoch + batch_idx

    # warmup 阶段用线性增长
    if current_step < warmup_steps:
        return BASE_LR * current_step / warmup_steps

    # cosine 衰减
    progress = (current_step - warmup_steps) / (
        (EPOCHS - WARMUP_EPOCHS) * len_epoch
    )
    return 0.5 * BASE_LR * (1 + math.cos(math.pi * progress))


def train_one_epoch(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        epoch: int,
        len_epoch: int
    ) -> tuple[float, float]:

    model.train()
    losses, top1 = AverageMeter(), AverageMeter()

    for step, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(DEVICE, non_blocking=True)  # 异步非阻塞
        targets = targets.to(DEVICE, non_blocking=True)

        # 动态设置学习率
        lr = get_lr(epoch, step, len_epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()

        # AMP 混合精度
        with autocast(device_type=DEVICE, enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = accuracy(outputs, targets)[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc, inputs.size(0))

        if step % 50 == 0:
            logging.info(
                f"- Train epoch [{epoch}], step {step}/{len_epoch} "
                f"Lr: {lr:.6f} Loss: {losses.avg:.4f} Accuracy: {top1.avg:.2f}%"
            )

    return losses.avg, top1.avg, lr


@torch.no_grad()

def validate(model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:

    model.eval()
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    for step, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        with autocast(device_type=DEVICE, enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1, inputs.size(0))
        top5.update(acc5, inputs.size(0))

    return losses.avg, top1.avg, top5.avg


def main():
    # ===== 命令行获取参数 =====
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", type=str, default=SPLIT)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="断点续训用的checkpoint路径，不指定则从头开始训练"
    )
    parser.add_argument(
        "--model-type", type=str, default="resnet50",
        choices=["resnet50", "4_SE", "3SE_1incep"],
        help="可选模型类型"
    )
    args = parser.parse_args()

    # ===== 设置对应实验路径 =====
    logs_dir = LOGS / Path(args.model_type)
    checkpoints_dir = CHECKPOINTS / Path(args.model_type)
    tensorboard_dir = logs_dir / "tensorboard"
    results_dir = RESULTS / Path(args.model_type)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(logs_dir)
    set_seed()

    # ===== 获取训练数据 =====
    train_loader, val_loader = get_dataloader(Path(args.split_json))
    len_epoch = len(train_loader)
    model = get_model(args.model_type)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=BASE_LR,
        momentum=0.9,
        nesterov=True,
        weight_decay=WEIGHT_DECAY
    )
    scaler = GradScaler(device=DEVICE, enabled=True)

    # ===== 数据初始化 =====
    start_epoch = 0
    best_acc = 0.0
    history = {
        "train_loss": [], "train_acc1": [],
        "val_loss": [], "val_acc1": [], "val_acc5": []
    }

    # ===== 断点续训 =====
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        history = checkpoint["history"]
        logging.info(f"从 {args.resume} 恢复训练，将从 epoch {start_epoch} 继续")

    writer = SummaryWriter(tensorboard_dir)

    # ===== 训练循环 =====
    logging.info("--------- 训练开始 ---------\n")
    for epoch in range(start_epoch, EPOCHS):
        logging.info(f"========= 第 {epoch + 1} 个 epoch =========")
        train_loss, train_acc1, lr = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, len_epoch
        )
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion
        )

        logging.info(
            f"\n[Epoch {epoch}] Loss: {train_loss:.4f} Accuracy: {train_acc1:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc1: {val_acc1:.2f}% Acc5: {val_acc5:.2f}%"
            "\n======================================\n"
        )

        history["train_loss"].append(train_loss)
        history["train_acc1"].append(train_acc1)
        history["val_loss"].append(val_loss)
        history["val_acc1"].append(val_acc1)
        history["val_acc5"].append(val_acc5)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.add_scalars("Accuracy",{"train": train_acc1, "val": val_acc1},  epoch + 1)
        writer.add_scalars(
            "Accuracy/val_top1&top5", {"val_acc1": val_acc1, "val_acc5": val_acc5}, epoch + 1
        )
        writer.add_scalar("Learning Rate", lr, epoch + 1)

        # ===== 最高 top1_acc =====
        is_best = val_acc1 > best_acc
        if is_best:
            best_acc = val_acc1

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "history": history
        }
        torch.save(state, checkpoints_dir / "latest.pth")
        if is_best:
            torch.save(state, checkpoints_dir / "best.pth")
            logging.info(f"- 最佳模型更新，val_acc1: {best_acc:.2f}%")
        if (epoch + 1) % 10 == 0:
            torch.save(state, checkpoints_dir / f"epoch_{epoch + 1}.pth")

    # ===== 保存训练历史 =====
    with open(results_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f)
    logging.info(f"训练完成，最佳 val_acc1: {best_acc:.2f}%")

    writer.close()


if __name__ == "__main__":
    main()