"""Net1D из ECGFounder.

Источник: https://github.com/PKUDigitalHealth/ECGFounder net1d.py
Лицензия MIT, Copyright (c) 2025 PKUDigitalHealth.
Архитектура Shenda Hong, март 2020. Здесь оставлен только класс сети.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        padding = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        left = padding // 2
        x = F.pad(x, (left, padding - left), "constant", 0)
        return self.conv(x)


class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = max(0, self.kernel_size - 1)
        left = padding // 2
        x = F.pad(x, (left, padding - left), "constant", 0)
        return self.max_pool(x)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: int,
        kernel_size: int,
        stride: int,
        groups: int,
        downsample: bool,
        is_first_block: bool = False,
        use_bn: bool = True,
        use_do: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.stride = stride if downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        middle = int(out_channels * ratio)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(in_channels, middle, kernel_size=1, stride=1, groups=1)

        self.bn2 = nn.BatchNorm1d(middle)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(middle, middle, kernel_size=kernel_size, stride=self.stride, groups=groups)

        self.bn3 = nn.BatchNorm1d(middle)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(middle, out_channels, kernel_size=1, stride=1, groups=1)

        squeeze = max(1, out_channels // 2)
        self.se_fc1 = nn.Linear(out_channels, squeeze)
        self.se_fc2 = nn.Linear(squeeze, out_channels)
        self.se_activation = Swish()
        if downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
            out = self.conv1(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do:
            out = self.do3(out)
        out = self.conv3(out)

        scale = out.mean(-1)
        scale = self.se_fc2(self.se_activation(self.se_fc1(scale)))
        out = torch.einsum("abc,ab->abc", out, torch.sigmoid(scale))

        if self.downsample:
            identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            extra = self.out_channels - self.in_channels
            left = extra // 2
            identity = F.pad(identity, (left, extra - left), "constant", 0)
            identity = identity.transpose(-1, -2)
        return out + identity


class BasicStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: int,
        kernel_size: int,
        stride: int,
        groups: int,
        i_stage: int,
        m_blocks: int,
        use_bn: bool = True,
        use_do: bool = True,
    ):
        super().__init__()
        self.block_list = nn.ModuleList()
        for i_block in range(m_blocks):
            is_first = i_stage == 0 and i_block == 0
            downsample = i_block == 0
            block_stride = stride if downsample else 1
            block_in = in_channels if downsample else out_channels
            self.block_list.append(
                BasicBlock(
                    in_channels=block_in,
                    out_channels=out_channels,
                    ratio=ratio,
                    kernel_size=kernel_size,
                    stride=block_stride,
                    groups=groups,
                    downsample=downsample,
                    is_first_block=is_first,
                    use_bn=use_bn,
                    use_do=use_do,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)
        return x


class Net1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_filters: int,
        ratio: int,
        filter_list: list[int],
        m_blocks_list: list[int],
        kernel_size: int,
        stride: int,
        groups_width: int,
        n_classes: int,
        use_bn: bool = True,
        use_do: bool = True,
    ):
        super().__init__()
        self.use_bn = use_bn
        self.first_conv = MyConv1dPadSame(in_channels, base_filters, kernel_size=kernel_size, stride=2)
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()
        self.stage_list = nn.ModuleList()
        channels = base_filters
        for index, (out_channels, blocks) in enumerate(zip(filter_list, m_blocks_list)):
            self.stage_list.append(
                BasicStage(
                    in_channels=channels,
                    out_channels=out_channels,
                    ratio=ratio,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=out_channels // groups_width,
                    i_stage=index,
                    m_blocks=blocks,
                    use_bn=use_bn,
                    use_do=use_do,
                )
            )
            channels = out_channels
        self.dense = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.first_conv(x)
        if self.use_bn:
            out = self.first_bn(out)
        out = self.first_activation(out)
        for stage in self.stage_list:
            out = stage(out)
        return self.dense(out.mean(-1))


def build_ecgfounder(n_classes: int = 150) -> Net1D:
    return Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=n_classes,
        use_bn=False,
        use_do=False,
    )
