"""Небольшая 1D CNN: [B, 12, 5000] → 24 логита. Сигмоида снаружи, в loss."""

from __future__ import annotations

import torch
from torch import nn


class RawECGCNN(nn.Module):
    def __init__(self, channels: int = 12, num_classes: int = 24) -> None:
        super().__init__()
        self.channels = channels
        self.features = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != self.channels:
            raise ValueError(f"Ожидался тензор [B, {self.channels}, N], получено {tuple(waveform.shape)}.")
        hidden = self.features(waveform).flatten(1)
        return self.head(hidden)
