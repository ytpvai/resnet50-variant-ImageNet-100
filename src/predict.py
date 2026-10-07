import argparse
import torch
import json
from torch import nn
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from config import *
import plot_style
from model import load_pretrained

plot_style.apply_style()


def predict_single_image(
        model: nn.Module, image_path: Path,
        classes: list[str], save_path: Path
    ):

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])
    image = Image.open(image_path).convert("RGB")
    input_img = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(input_img)
        probs = torch.softmax(output, dim=1)[0]
        top5_probs, top5_indices = probs.topk(5)

    print(f"图片{image_path}预测结果：")
    for i, (prob, idx) in enumerate(
        zip(top5_probs.to("cpu").numpy(), top5_indices.to("cpu").numpy())
    ):
        print(f"  [{i + 1}] {classes[idx]:20s}  {prob * 100:.2f}%")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].barh(range(5), top5_probs.cpu().numpy()[::-1],
                 color=plot_style.PALETTE[0])
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(
        [classes[idx] for idx in top5_indices.cpu().numpy()[::-1]], fontsize=9
    )
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Top-5 Predictions")
    axes[1].set_xlim(0, 1)

    fig.tight_layout()
    plot_style.save(fig, save_path)
    print(f"预测结果图在：{save_path}")
    plt.close()

    return classes[top5_indices[0].item()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type", type=str, default="resnet50",
        choices=["resnet50", "4_SE", "3SE_1incep", "resnext50"],
        help="可选模型类型"
    )
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split-json", type=str, default=SPLIT)
    args = parser.parse_args()

    figures_dir = RESULTS / Path(args.model_type) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split_json, "r") as f:
        classes = json.load(f)["class_names"]

    if not Path(args.checkpoint).exists():
        print("此路径不存在，请先运行: python train.py")
        return
    model, _ = load_pretrained(args.model_type, args.checkpoint)
    print(f"已加载模型{args.model_type}")

    if args.image:
        predict_single_image(
            model, Path(args.image), classes, figures_dir / f"{Path(args.image).stem}.png"
        )
    else:
        print("请用 --image 指定图片路径")


if __name__ == "__main__":
    main()