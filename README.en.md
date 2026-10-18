# Training ResNet-50 and Its Variants from Scratch on ImageNet-100

**English** | [简体中文](README.md)

We train ResNet-50 and three improved variants from scratch on ImageNet-100 (100 classes) and evaluate them on the held-out official validation set (5000 images). The best variant, `4_SE` (channel attention added to every Bottleneck), reaches **87.82%** test top-1, against 85.30% for the baseline.

The code is built on PyTorch. Except for some Windows-compatibility and visualization code written by AI, everything else was implemented by the author.

## Results

| Model | Architecture | Params | Test top-1 (%) | 95% CI | Test top-5 (%) |
|---|---|---|---|---|---|
| `resnet50` | baseline | 23.7M | 85.30 | 84.3 – 86.3 | 97.18 |
| `resnext50` | ResNeXt-50 (32×4d) + channel attention | 33.3M | 87.16 | 86.2 – 88.1 | 97.78 |
| `3SE_1incep` | channel attention in layer1–3, Inception + SE in layer4 | 34.9M | 86.68 | 85.7 – 87.6 | 97.64 |
| `4_SE` | channel attention in all 16 blocks | 33.8M | **87.82** | 86.9 – 88.7 | **97.86** |

- The test set has 5000 images (50 per class). The 95% confidence intervals use a binomial approximation and reflect test-set sampling error only; they do not account for training randomness (each model was trained once).
- Parameter counts are computed from the model architecture (100-way classifier) and exclude BN running statistics.

![Test top-1 comparison](results/test_top1.png)

### Reading the results

1. All three variants beat the baseline (+1.4 to +2.5 points), and the gap between `4_SE` and the baseline is the clearest.
2. `4_SE`, `resnext50` and `3SE_1incep` differ from one another by no more than 1.14 points, well within their respective confidence intervals; **we cannot tell which of them is better**.
3. Replacing layer4 with an Inception + SE block (whose SE is a single-path, average-pooling-only version) brought no additional gain: the point estimate for `3SE_1incep` is 1.14 points below `4_SE` (not significant). That difference conflates two changes at once—the multi-branch Inception structure and the different attention form—so it cannot be attributed to Inception alone.
4. **All variants have roughly 40% more parameters than the baseline**, mostly from channel attention with `reduction=4` (see below). This project therefore cannot separate gains coming from the attention mechanism itself from gains coming from larger model capacity, and no parameter-matched control was run.
5. Pairwise correlations between per-class accuracies of the four models are 0.95 – 0.97; errors concentrate on the same set of fine-grained neighbouring classes. Changing the block structure did not change "which classes are hard".

Confusion matrix for `4_SE` (100×100; the color scale caps at 50, i.e. the number of samples per class):

![4_SE confusion matrix](results/4_SE/figures/confusion_matrix.png)


## Model architecture

Shared by all four models:

- Stem: three 3×3 convolutions (32 – 32 – 64, the first with stride 2) + BN + MaxPool (ResNet-C style).
- Block counts per stage are [3, 4, 6, 3], with the stride placed on the 3×3 convolution (ResNet-B).
- Downsample branch: AvgPool(stride) + 1×1 conv + BN (ResNet-D style).
- Input 224×224, global average pooling followed by a 2048 → 100 fully connected layer.

Differences between the variants:

Per-stage specification (identical for all four models, input 224×224):

| Stage | Blocks | mid channels | Output channels | Output size |
|---|---|---|---|---|
| stem (three 3×3 + MaxPool) | — | — | 64 | 56×56 |
| layer1 | 3 | 64 | 256 | 56×56 |
| layer2 | 4 | 128 | 512 | 28×28 |
| layer3 | 6 | 256 | 1024 | 14×14 |
| layer4 | 3 | 512 | 2048 | 7×7 |

Block type used at each stage, per model:

| Model | layer1 – 3 | layer4 |
|---|---|---|
| `resnet50` | Bottleneck | Bottleneck |
| `4_SE` | SEBottleneck | SEBottleneck |
| `3SE_1incep` | SEBottleneck | InceptionResBlock |
| `resnext50` | ResNeXtBottleneck | ResNeXtBottleneck |

`3SE_1incep` and `4_SE` differ only in the 3 blocks of layer4 and are otherwise identical; `resnet50` and `resnext50` differ from `4_SE` in all 16 blocks.

### Inside the blocks

Let `in` be the input channels, `mid` the stage's mid channels, and `4·mid` the block output channels. In every block the stride sits on the 3×3 convolution, and the downsampling shortcut is AvgPool(stride) + 1×1 conv + BN; when stride is 1 but the channel count changes (the first block of layer1) it is just 1×1 conv + BN.

**Bottleneck (`resnet50`)**
- Main path: 1×1 conv (in→mid) → BN → ReLU → 3×3 conv (mid→mid, stride) → BN → ReLU → 1×1 conv (mid→4·mid) → BN
- Added to the shortcut, then ReLU

**SEBottleneck (`4_SE`, and layer1 – 3 of `3SE_1incep`)**
- Main path as in Bottleneck, with channel attention inserted after the final BN and before the addition with the shortcut
- Channel attention: global average pooling and global max pooling of the feature, both passed through the *same* shared MLP (1×1 conv, C→C/4, ReLU, 1×1 conv, C/4→C); the two outputs are summed, passed through a sigmoid, and the resulting channel weights are multiplied back into the feature

**InceptionResBlock (layer4 of `3SE_1incep`)**
- With b = mid/2, three parallel branches all take the input:
  - Branch 1: AvgPool(stride) → 1×1 conv (in→b) → BN → ReLU
  - Branch 2: 1×1 conv (in→b) → BN → ReLU → 3×3 conv (b→b, stride) → BN → ReLU
  - Branch 3: 1×1 conv (in→b) → BN → ReLU → 3×3 conv (b→b) → BN → ReLU → 3×3 conv (b→b, stride) → BN → ReLU
- The three branches are concatenated along channels (3b channels) and projected by 1×1 conv (3b→4·mid) + BN
- Channel attention follows (here a single-path version using average pooling only, `reduction` 4), then addition with the shortcut and ReLU

**ResNeXtBottleneck (`resnext50`)**
- 1×1 conv (in→2·mid) → BN → ReLU → 3×3 grouped conv (2·mid→2·mid, groups=32, stride) → BN → ReLU → 1×1 conv (2·mid→4·mid) → BN
- Group width (channels per group) is 4, 8, 16 and 32 for layer1 – 4 respectively
- Followed by the same channel attention as `SEBottleneck`, then addition with the shortcut and ReLU

Notes on the channel-attention implementation:

- `SEModule` applies both global average pooling and global max pooling to the feature and passes them through **the same** shared two-layer 1×1-conv MLP (`reduction = 4`); the two outputs are summed and passed through a sigmoid to produce the channel weights. This differs from the original SE, which uses average pooling only, and is closer to the channel attention module of CBAM.
- The SE inside `InceptionResBlock` is single-path (average pooling only).
- `reduction = 4` is larger than the more common 16. Under this project's architecture, channel attention adds 10.08M parameters to `4_SE` (+42.5%); with 16 the increase would be about 2.53M (+10.7%). All results here use `reduction = 4`, and no ablation on it was run.


## Training setup

| Item | Setting |
|---|---|
| Data | ImageNet-100; the training pool is split 95% / 5% into train / val (seed 42) |
| Input size | 224 × 224 |
| Batch size | 80 |
| Optimizer | SGD, momentum 0.9, Nesterov |
| Learning rate | Peak 0.1 × 80 / 256 = 0.03125; 5 epochs of linear warmup, then step-wise cosine annealing |
| Weight decay | 5e-4 |
| Label smoothing | 0.1 |
| Epochs | 200 |
| Mixed precision | AMP |
| Data augmentation | RandomResizedCrop, horizontal flip, RandAugment (N = 2, M = 12), RandomErasing (p = 0.25), as defined by `build_transforms` in `src/utils.py`; no Mixup / CutMix |
| Model selection | The epoch with the highest val top-1 is saved as `best.pth` |

Evaluation protocol:

- val is used only for training monitoring and for selecting `best.pth`.
- The official validation set (`val.X`) was renamed to `data/test` and is used for the final evaluation only; it contains 50 images per class.


## Quick start

Environment: Python 3.10, PyTorch 2.12.0 (CUDA 13.0 build), a single NVIDIA GeForce RTX 5050 Laptop GPU (8 GB).

Non-PyTorch dependencies:

```bash
pip install -r requirements.txt
```

torch / torchvision must be CUDA builds, and the Windows builds on PyPI are CPU-only, so install them separately from the official PyTorch index:

```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
```

### 1. Prepare the data

1. Download ImageNet-100 from Kaggle and comply with the dataset page's and ImageNet's terms of use. This repository does not provide the data.
2. Place it under `data/`: merge `train.X1` – `train.X4` into `data/train/<synset>/*.JPEG`, rename `val.X` to `data/test/`, and keep `Labels.json`.
3. Generate the train / val split (only needs to be run once):

```bash
python src/data_split.py     # writes data/split.json
```

### 2. Training

```bash
python src/train.py --model-type 4_SE
# Available: resnet50 / 4_SE / 3SE_1incep / resnext50
# Resume from a checkpoint: --resume checkpoints/4_SE/latest.pth
```

Logs are written to `logs/`, checkpoints to `checkpoints/<model-type>/`, and the training history to `results/<model-type>/history.json`. To watch the training curves live:

```bash
tensorboard --logdir logs
```

### 3. Evaluation

```bash
python src/evaluate.py --model-type 4_SE
```

Output goes to `results/<model-type>/`:

- `test_metrics.json`: top-1, top-5, standard error, per-class accuracy.
- `test_outputs.npz`: per-sample labels, predictions, probabilities, the confusion matrix and image paths, usable for paired significance tests.
- `figures/confusion_matrix.png`: the confusion matrix.


## Project structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py        # paths and hyperparameters
│   ├── data_split.py    # train / val split
│   ├── evaluate.py      # test-set evaluation
│   ├── model.py         # the four models and checkpoint loading
│   ├── plot_style.py    # plotting style settings
│   ├── predict.py       # single-image prediction
│   ├── train.py         # training
│   ├── utils.py         # data loading, augmentation, AverageMeter, accuracy
│   └── visualize.py     # plotting
├── results/
│   ├── test_top1.png          # test top-1 comparison of the four models
│   └── <model>/               # history.json, test_metrics.json, test_outputs.npz, figures/
├── data/                      # not shipped with the repository
├── checkpoints/               # not shipped with the repository
└── logs/                      # not shipped with the repository (<model>/tensorboard/ holds TensorBoard event files)
```


## Limitations

- **Each model was trained only once**, so seed-to-seed variation was not measured. Gaps smaller than roughly 1 point between models should not be treated as conclusions.
- The test set has only 5000 images; the 95% confidence interval of a single model's top-1 is about ±0.9 points.
- **Mismatched parameter counts**: the three variants have about 40% more parameters than the baseline, so part of the improvement may come from extra capacity rather than from the architectural design alone.
- The four candidates were compared against their test results, which introduces a mild selection bias; strictly speaking, the selection should be based on val.
- For the two models with training logs, val scores are about 3 points higher than test; we did not investigate why, so the table above reports test only.


## References

- K. He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- T. He et al. *Bag of Tricks for Image Classification with Convolutional Neural Networks.* CVPR 2019.
- J. Hu et al. *Squeeze-and-Excitation Networks.* CVPR 2018.
- S. Woo et al. *CBAM: Convolutional Block Attention Module.* ECCV 2018.
- S. Xie et al. *Aggregated Residual Transformations for Deep Neural Networks.* CVPR 2017.
- C. Szegedy et al. *Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning.* AAAI 2017.
- E. D. Cubuk et al. *RandAugment: Practical Automated Data Augmentation with a Reduced Search Space.* CVPRW 2020.
- Z. Zhong et al. *Random Erasing Data Augmentation.* AAAI 2020.
