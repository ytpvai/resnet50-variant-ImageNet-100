import torch
from torch import nn
from pathlib import Path
from config import NUM_CLASSES, DEVICE, CHECKPOINTS


# ===== 基础 Bottleneck 块 =====

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample: nn.Module | None=None):
        super().__init__()

        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


# ===== SE Bottleneck 块 =====

class SEModule(nn.Module):
    """SE 模块"""

    reduction = 4

    def __init__(self, channels, reduction=reduction):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        avg_s = self.mlp(self.avgpool(x))
        max_s = self.mlp(self.maxpool(x))
        s = self.sigmoid(avg_s + max_s)
        return x * s

class SEBottleneck(Bottleneck):

    reduction = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None, reduction=reduction):
        super().__init__(in_channels, mid_channels, stride, downsample)
        out_channels = mid_channels * self.expansion
        self.se = SEModule(out_channels, reduction=reduction)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# ===== 混合 Inception 块 =====

class InceptionResBlock(nn.Module):
    """Inception, ResNet, SE 混合"""

    expansion = 4
    reduction = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion
        branch_channels = mid_channels // 2 # 经计算此值可使参数量与ResNet的block相差不大

        self.branch1 = nn.Sequential(
            # 减少信息丢失
            nn.AvgPool2d(kernel_size=stride, stride=stride,
                         ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // self.reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // self.reduction, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.project = nn.Conv2d(branch_channels * 3, out_channels, kernel_size=1, bias=False)
        self.bn_project = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1) # 注意按通道合
        out = self.bn_project(self.project(out))
        out = self.se(out) * out # 注意权重的广播
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNeXtBottleneck(nn.Module):
    """ ResNeXt 分组卷积 """

    expansion = 4
    cardinality = 32

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion
        grouped_channels = mid_channels * 2
        assert grouped_channels % self.cardinality == 0, \
            f"分组通道数({grouped_channels})必须能被cardinality({self.cardinality})整除"

        self.conv1 = nn.Conv2d(in_channels, grouped_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(grouped_channels)
        self.conv2 = nn.Conv2d(grouped_channels, grouped_channels, kernel_size=3,
                               stride=stride, padding=1, groups=self.cardinality,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(grouped_channels)
        self.conv3 = nn.Conv2d(grouped_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SEModule(out_channels, reduction=4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50(nn.Module):
    def __init__(self, block_cls: nn.Module | list[nn.Module] =Bottleneck,
                 legacy_downsample: bool = False):
        super().__init__()

        # 旧版 checkpoint（ResNet-D 改动之前训练的）的 downsample 是 Conv(stride)+BN，
        # 加载这类权重时需要按旧结构建模型，否则 state_dict 对不上
        self.legacy_downsample = legacy_downsample
         # 统一转换成一个长度为4的列表
        if isinstance(block_cls, (list, tuple)):
            assert len(block_cls) == 4, "block_cls传列表时，必须正好是4个元素，对应layer1~4"
            block_classes = list(block_cls)
        # 或只传一个
        else:
            block_classes = [block_cls] * 4

        self.in_channels = 64

        # stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个 layer
        self.layer1 = self._make_layer(64, 3, 1, block_cls=block_classes[0])
        self.layer2 = self._make_layer(128, 4, 2, block_cls=block_classes[1])
        self.layer3 = self._make_layer(256, 6, 2, block_cls=block_classes[2])
        self.layer4 = self._make_layer(512, 3, 2, block_cls=block_classes[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_classes[3].expansion, NUM_CLASSES)

        self._initialize_weights()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _make_layer(self, mid_channels, blocks, stride, block_cls: nn.Module=Bottleneck):
        """用 Bottleneck 构建 layer"""

        downsample = None
        out_channels = mid_channels * block_cls.expansion
        if stride != 1 or self.in_channels != out_channels:

            # 新版（ResNet-D）：AvgPool 前置做空间下采样，减少 stride 在 1*1 卷积上的信息损失
            # 旧版（legacy）：直接 Conv(stride)+BN，与早期训练的 checkpoint 结构一致
            temp_layers = []
            if stride != 1 and not self.legacy_downsample:
                temp_layers.append(nn.AvgPool2d(
                        kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False
                )) # ceil_mode=True 补充虚拟 0，需 c_i_p 除去其影响
            conv_stride = 1 if (stride != 1 and not self.legacy_downsample) else stride
            temp_layers.append(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                          stride=conv_stride, bias=False)
            )
            temp_layers.append(nn.BatchNorm2d(out_channels))
            downsample = nn.Sequential(*temp_layers)

        layers = [block_cls(self.in_channels, mid_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block_cls(self.in_channels, mid_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """权重初始化"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)


def get_model(model_type="resnet50", legacy_downsample=False):
    if model_type == "resnet50":
        block_cls = Bottleneck
    elif model_type == "4_SE":
        block_cls = SEBottleneck
    elif model_type == "3SE_1incep":
        block_cls = [SEBottleneck, SEBottleneck, SEBottleneck, InceptionResBlock]
    elif model_type == "resnext50":
        block_cls = ResNeXtBottleneck   # 32x4d 加宽在 block 内部实现
    else:
        raise ValueError(f"未知的model_type: {model_type}")
    model = ResNet50(block_cls, legacy_downsample=legacy_downsample)

    return model.to(DEVICE)


def load_pretrained(model_type="resnet50", checkpoint_path=None):
    """构建与 checkpoint 匹配的模型并加载权重，返回 (model, checkpoint)。

    历史原因：ResNet-D 改动之前训练的 checkpoint（如旧版 resnet50）的
    downsample 是 Conv(stride)+BN，之后的是 AvgPool+Conv+BN。
    这里通过检查 state_dict 里的关键key自动判断该用哪种结构，
    避免结构对不上导致 load_state_dict 报错。
    """
    ckpt_path = Path(checkpoint_path) if checkpoint_path \
        else CHECKPOINTS / model_type / "best.pth"
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    state = checkpoint["model_state_dict"]

    # 新版 downsample 是3层 Sequential，BN 在 index 2；旧版2层，没有 index 2
    legacy = "layer2.0.downsample.2.weight" not in state
    model = get_model(model_type, legacy_downsample=legacy)
    model.load_state_dict(state)
    return model, checkpoint


if __name__ == "__main__":
    model = ResNet50()
    x = torch.randn(2, 3, 224, 224)
    y = get_model()(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")   # 期望: torch.Size([2, 100])
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")