# ImageNet-100 · ResNet-50 项目

本项目每个文件都拆开保存，注释比之前那份合并文档更详细——尤其是之前讲得比较
简略的 AMP（autocast/GradScaler）、cosine warmup 学习率、Bottleneck 设计原理
这几块，这次都补充了完整的原理说明。

## 文件依赖关系（先看这个，理解执行顺序）

```
config.py  ←── 被所有其他文件 import，全局参数唯一来源
    │
    ├── plot_style.py   （被 visualize.py / predict.py import，管画图风格）
    │
    ├── data_split.py   【第一步，单独执行一次】
    │       生成 split.json（train/val 切分 + 类别名映射）
    │           │
    │           ▼
    ├── model.py         （被 train / evaluate / visualize / predict import）
    ├── utils.py          （被 train / evaluate / visualize import）
    │           │
    │           ▼
    ├── train.py         【第二步，执行训练】
    │       读 split.json → 训练 → 输出 checkpoints/*.pth + history
    │           │
    │           ▼
    ├── evaluate.py       【第三步，训练全部结束后，只跑一次】
    │       读 checkpoints/best.pth + 官方 val.X(=test) → 输出最终指标
    │       调用 visualize.py 的 plot_confusion_matrix 画图
    │
    ├── visualize.py       （被 evaluate.py 调用，也可单独用命令行跑三种模式）
    └── predict.py          （独立使用，对单张图片做预测）
```

## 运行顺序

```bash
# 1. 只需跑一次，生成数据切分
python data_split.py

# 2. 训练（时间会比较长）
python train.py

# 3. 训练过程中随时可以看训练曲线
python visualize.py --mode history

# 4. 看看数据集/特征图长什么样（可选，随时能跑）
python visualize.py --mode dataset
python visualize.py --mode features --checkpoint checkpoints/best.pth

# 5. 训练彻底结束、不再调整任何配置之后，只跑一次
python evaluate.py --checkpoint checkpoints/best.pth

# 6. 对自己找的图片做预测
python predict.py --image my_photo.jpg --checkpoint checkpoints/best.pth
```

## 还留给你自己实现的部分（按优先级）

1. `data_split.py` — 里面 `archive_dir` 遍历逻辑要对应你合并后的实际目录结构
2. `utils.py` 里的 `mixup_cutmix_batch` — 全项目唯一留空的核心逻辑，函数内的
   注释里写了完整的实现思路，建议先单独写个小测试脚本验证形状和数值再接入训练
3. 确认 `config.DATA_ROOT` 指向你实际的路径（或者符号链接）

其余部分（尤其是 model.py、plot_style.py、train.py 的训练循环骨架）都是完整
可运行的代码，可以直接用。卡在哪一块，随时把代码贴给我。
