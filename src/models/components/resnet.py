import torch.nn as nn
from .lora_layers import Conv2d

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if lora:
            self.conv1 = Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
            )
        self.bn1 = nn.BatchNorm2d(out_channels)

        if lora:
            self.conv2 = Conv2d(
                out_channels, out_channels, kernel_size=3, stride=stride, padding=1, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
        else:
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if lora:
            self.conv3 = Conv2d(
                out_channels,
                out_channels * self.expansion,
                kernel_size=1,
                stride=1,
                padding=0,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
        else:
            self.conv3 = nn.Conv2d(
                out_channels,
                out_channels * self.expansion,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.relu(self.bn2(self.conv2(x)))

        x = self.conv3(x)
        x = self.bn3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


# class Block(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
#         super(Block, self).__init__()

#         self.conv1 = Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             stride=stride,
#             bias=False,
#             r=5
#         )
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = Conv2d(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             stride=stride,
#             bias=False,
#             r=5
#         )
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.i_downsample = i_downsample
#         self.stride = stride
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         identity = x.clone()

#         x = self.relu(self.bn2(self.conv1(x)))
#         x = self.bn2(self.conv2(x))

#         if self.i_downsample is not None:
#             identity = self.i_downsample(identity)
#         x += identity
#         x = self.relu(x)
#         return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3, lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.lora = lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        if lora:
            self.conv1 = Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
        else:
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        # x = self.softmax(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            if self.lora:
                conv_layer = Conv2d(
                    self.in_channels,
                    planes * ResBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout
                )
            else:
                conv_layer = nn.Conv2d(
                    self.in_channels,
                    planes * ResBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                )
            ii_downsample = nn.Sequential(
                    conv_layer,
                    nn.BatchNorm2d(planes * ResBlock.expansion),
                )


        layers.append(
            ResBlock(
                self.in_channels, planes, i_downsample=ii_downsample, stride=stride
            )
        )
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet50(num_classes, channels=3, lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels, lora, lora_r, lora_alpha, lora_dropout)


def ResNet101(num_classes, channels=3, lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels, lora, lora_r, lora_alpha, lora_dropout)


def ResNet152(num_classes, channels=3, lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels, lora, lora_r, lora_alpha, lora_dropout)