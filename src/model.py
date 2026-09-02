import torch
from torch import nn
from config import NUM_CLASSES, DEVICE


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


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

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
        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, NUM_CLASSES)

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

    def _make_layer(self, mid_channels, blocks, stride):
        """用 Bottleneck 构建 layer"""

        downsample = None
        out_channels = mid_channels * Bottleneck.expansion
        if stride != 1 or self.in_channels != out_channels:

            # 加入 avgpool 减少信息在 stride 非 1 时在 1*1 卷积的损失
            temp_layers = []
            if stride != 1:
                temp_layers.append(nn.AvgPool2d(
                        kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False
                )) # ceil_mode=True 补充虚拟 0，需 c_i_p 除去其影响
            temp_layers.append(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            downsample = nn.Sequential(*temp_layers)

        layers = [Bottleneck(self.in_channels, mid_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, mid_channels))

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


def get_model(model_type="resnet50"):
    if model_type == "resnet50":
        model = ResNet50()
    else:
        raise ValueError(f"未知的model_type: {model_type}")
    return model.to(DEVICE)


if __name__ == "__main__":
    model = ResNet50()
    x = torch.randn(2, 3, 224, 224)
    y = get_model()(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")   # 期望: torch.Size([2, 100])
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")