# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Research detection head modules for optional YAML-level switching.

The heads in this file are intentionally not wired into any default model. They are Detect-compatible modules that can
be selected from a model YAML when running ablation experiments.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv
from .head import Detect
from .transformer import AIFI

__all__ = (
    "ASFFHead",
    "AdaptiveHead",
    "LADH",
    "SERDet",
    "TPHDetect",
    "TransformerSmallDetect",
)


def _make_head_end2end(head: Detect, end2end: bool) -> None:
    """Attach one-to-one branches after a Detect subclass replaces its default heads."""
    head.end2end = end2end
    if end2end:
        head.one2one_cv2 = copy.deepcopy(head.cv2)
        head.one2one_cv3 = copy.deepcopy(head.cv3)


def _attention_heads(channels: int, requested: int = 8) -> int:
    """Return a valid multi-head attention count for a channel dimension."""
    return max(x for x in range(min(requested, channels), 0, -1) if channels % x == 0)


class _PointwiseHead(nn.Sequential):
    """Two 1x1 Conv blocks followed by a prediction conv."""

    def __init__(self, c1: int, c2: int, out_channels: int):
        super().__init__(Conv(c1, c2, 1), Conv(c2, c2, 1), nn.Conv2d(c2, out_channels, 1))


class _DepthwisePointwiseHead(nn.Sequential):
    """Repeated depthwise-separable 3x3 blocks followed by a prediction conv."""

    def __init__(self, c1: int, c2: int, out_channels: int, repeats: int = 2):
        blocks = []
        c = c1
        for _ in range(repeats):
            blocks.append(nn.Sequential(DWConv(c, c, 3), Conv(c, c2, 1)))
            c = c2
        super().__init__(*blocks, nn.Conv2d(c2, out_channels, 1))


class _AsymmetricRegressionHead(nn.Sequential):
    """LADH regression branch with asymmetric multi-level channel compression."""

    def __init__(self, c1: int, c2: int, out_channels: int):
        c3, c4 = max(c2 // 2, 16), max(c2 // 4, 16)
        super().__init__(
            nn.Sequential(DWConv(c1, c1, 3), Conv(c1, c2, 1)),
            nn.Sequential(DWConv(c2, c2, 3), Conv(c2, c3, 1)),
            nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c4, 1)),
            nn.Conv2d(c4, out_channels, 1),
        )


class _TransformerPredictionBranch(nn.Sequential):
    """Transformer prediction branch used by TPH-YOLOv5-style heads."""

    def __init__(self, c1: int, c2: int, out_channels: int, heads: int = 8, dropout: float = 0.0):
        heads = _attention_heads(c2, heads)
        super().__init__(
            Conv(c1, c2, 1),
            AIFI(c2, cm=c2 * 4, num_heads=heads, dropout=dropout, normalize_before=False),
            Conv(c2, c2, 3),
            nn.Conv2d(c2, out_channels, 1),
        )


class TPHDetect(Detect):
    """Transformer Prediction Head from TPH-YOLOv5 for drone/small-object detection.

    Paper: "TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on
    Drone-Captured Scenarios" (ICCVW 2021). The head replaces convolutional prediction branches with transformer
    encoder blocks and supports an extra high-resolution feature level when the YAML supplies one.
    """

    def __init__(
        self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = (), heads: int = 8, dropout: float = 0.0
    ):
        """Initialize a TPH-YOLOv5-style Detect-compatible head."""
        super().__init__(nc, reg_max, end2end=False, ch=ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            _TransformerPredictionBranch(x, c2, 4 * self.reg_max, heads=heads, dropout=dropout) for x in ch
        )
        self.cv3 = nn.ModuleList(_TransformerPredictionBranch(x, c3, self.nc, heads=heads, dropout=dropout) for x in ch)
        _make_head_end2end(self, end2end)


class TransformerSmallDetect(TPHDetect):
    """Alias with an explicit small-object name for YAML readability."""


class LADH(Detect):
    """Lightweight Asymmetric Detection Head.

    Paper source: "Faster and Lightweight: An Improved YOLOv5 Object Detector for Remote Sensing Images" describes
    LADH-Head as a lightweight asymmetric decoupled head: classification uses two 1x1 convolutions, while localization
    uses three depthwise-separable 3x3 convolutions plus a 1x1 prediction layer.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        """Initialize LADH with asymmetric classification and localization branches."""
        super().__init__(nc, reg_max, end2end=False, ch=ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0] // 2, min(self.nc, 100))
        self.cv2 = nn.ModuleList(_AsymmetricRegressionHead(x, c2, 4 * self.reg_max) for x in ch)
        self.cv3 = nn.ModuleList(_PointwiseHead(x, c3, self.nc) for x in ch)
        _make_head_end2end(self, end2end)


class SaELayer(nn.Module):
    """Split-enhance attention layer used by SERDet.

    The layer follows the paper's four-branch channel-attention description: squeeze global context, process it through
    parallel fully connected branches with different widths, concatenate the branch responses, and project back to
    per-channel weights.
    """

    def __init__(self, channels: int, reduction: int = 16):
        """Initialize the four-branch attention layer."""
        super().__init__()
        hidden = max(channels // reduction, 4)
        branch_widths = (hidden, hidden * 2, hidden * 3, hidden)
        self.pre = Conv(channels, channels, 1)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Linear(channels, width),
                nn.ReLU(inplace=True),
                nn.Linear(width, width),
                nn.ReLU(inplace=True),
            )
            for width in branch_widths
        )
        self.post = nn.Linear(sum(branch_widths), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply four-branch channel recalibration."""
        u = self.pre(x)
        pooled = F.adaptive_avg_pool2d(u, 1).flatten(1)
        weights = torch.cat([branch(pooled) for branch in self.branches], dim=1)
        weights = self.post(weights).sigmoid().view(x.shape[0], x.shape[1], 1, 1)
        return x * weights


class _SERBoxBranch(nn.Sequential):
    """SERDet regression branch: 3x3 conv, SaELayer, 3x3 conv, prediction conv."""

    def __init__(self, c1: int, c2: int, out_channels: int):
        super().__init__(Conv(c1, c2, 3), SaELayer(c2), Conv(c2, c2, 3), nn.Conv2d(c2, out_channels, 1))


class SERDet(Detect):
    """Selective Enhancement for Regression Detection head from SER-YOLOv8.

    Paper: "SER-YOLOv8: An Early Forest Fire Detection Model Integrating Multi-Path Attention and NWD" (Forests 2026).
    The classification branch is made lightweight with DWConv, while the regression branch uses SaELayer-based
    multi-path channel recalibration between progressive 3x3 feature extraction stages.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        """Initialize SERDet with a SaELayer regression branch and DWConv classification branch."""
        super().__init__(nc, reg_max, end2end=False, ch=ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(_SERBoxBranch(x, c2, 4 * self.reg_max) for x in ch)
        self.cv3 = nn.ModuleList(_DepthwisePointwiseHead(x, c3, self.nc, repeats=2) for x in ch)
        _make_head_end2end(self, end2end)


class ASFFBlock(nn.Module):
    """Adaptively Spatial Feature Fusion block.

    Paper: "Learning Spatial Fusion for Single-Shot Object Detection" (arXiv:1911.09516). Each target level resizes all
    pyramid inputs to the target spatial resolution, learns per-location softmax weights, and fuses features
    spatially before prediction.
    """

    def __init__(self, channels: tuple[int, ...], level: int, compress_channels: int = 16):
        """Initialize ASFF for one target feature-pyramid level."""
        super().__init__()
        self.level = level
        self.target_channels = channels[level]
        self.proj = nn.ModuleList(
            nn.Identity() if c == self.target_channels else Conv(c, self.target_channels, 1) for c in channels
        )
        self.weight = nn.ModuleList(Conv(self.target_channels, compress_channels, 1) for _ in channels)
        self.weight_levels = nn.Conv2d(compress_channels * len(channels), len(channels), 1)
        self.expand = Conv(self.target_channels, self.target_channels, 3)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Fuse a list of pyramid features into this block's target level."""
        target_size = features[self.level].shape[-2:]
        resized = []
        for feature, proj in zip(features, self.proj):
            feature = proj(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="nearest")
            resized.append(feature)

        weights = torch.cat([weight(feature) for weight, feature in zip(self.weight, resized)], dim=1)
        weights = self.weight_levels(weights).softmax(dim=1)
        fused = sum(feature * weights[:, i : i + 1] for i, feature in enumerate(resized))
        return self.expand(fused)


class ASFFHead(Detect):
    """Detect-compatible head with ASFF fusion before per-level prediction branches."""

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        compress_channels: int = 16,
    ):
        """Initialize ASFFHead."""
        super().__init__(nc, reg_max, end2end=False, ch=ch)
        self.asff = nn.ModuleList(ASFFBlock(ch, i, compress_channels=compress_channels) for i in range(len(ch)))
        _make_head_end2end(self, end2end)

    def _fuse_features(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply ASFF to every output level."""
        return [fusion(x) for fusion in self.asff]

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse features with ASFF, then run the standard Detect forward path."""
        return super().forward(self._fuse_features(x))


class _TaskAwareActivation(nn.Module):
    """Task-aware channel activation from Dynamic Head-style adaptive heads."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, channels * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two-piece task-aware activation."""
        context = F.adaptive_avg_pool2d(x, 1).flatten(1)
        alpha1, alpha2, beta1, beta2 = self.fc2(self.norm(F.relu(self.fc1(context), inplace=True))).chunk(4, dim=1)
        alpha1 = (alpha1.sigmoid() * 2 - 1).view(x.shape[0], x.shape[1], 1, 1)
        alpha2 = (alpha2.sigmoid() * 2 - 1).view(x.shape[0], x.shape[1], 1, 1)
        beta1 = (beta1.sigmoid() * 2 - 1).view(x.shape[0], x.shape[1], 1, 1)
        beta2 = (beta2.sigmoid() * 2 - 1).view(x.shape[0], x.shape[1], 1, 1)
        return torch.maximum(alpha1 * x + beta1, alpha2 * x + beta2)


class AdaptiveHeadBlock(nn.Module):
    """Adaptive Head feature block with scale, spatial, and task attention."""

    def __init__(self, channels: tuple[int, ...], level: int):
        super().__init__()
        self.level = level
        self.target_channels = channels[level]
        self.proj = nn.ModuleList(
            nn.Identity() if c == self.target_channels else Conv(c, self.target_channels, 1) for c in channels
        )
        self.scale_attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.target_channels, 1, 1), nn.Hardsigmoid())
        self.spatial_attn = nn.Sequential(DWConv(self.target_channels, self.target_channels, 3), nn.Sigmoid())
        self.task_attn = _TaskAwareActivation(self.target_channels)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Adapt one target feature level with scale, spatial, and task attention."""
        target_size = features[self.level].shape[-2:]
        fused = []
        scale_weights = []
        for feature, proj in zip(features, self.proj):
            feature = proj(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="nearest")
            fused.append(feature)
            scale_weights.append(self.scale_attn(feature))
        scale_weights = torch.stack(scale_weights, dim=1).softmax(dim=1)
        x = sum(feature * scale_weights[:, i] for i, feature in enumerate(fused))
        x = x * self.spatial_attn(x)
        return self.task_attn(x)


class AdaptiveHead(Detect):
    """Adaptive Head inspired by ADA-YOLO and Dynamic Head attention decomposition.

    Sources: ADA-YOLO (arXiv:2312.10099) describes an Adaptive Head that uses dynamic feature localization and guided
    parallel regression; its attention equations follow the Dynamic Head decomposition into scale-, spatial-, and
    task-aware attention. This implementation keeps the Ultralytics Detect interface unchanged and applies those
    adaptive attentions as a feature adapter before the standard regression/classification branches.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        """Initialize AdaptiveHead."""
        super().__init__(nc, reg_max, end2end=False, ch=ch)
        self.adapt = nn.ModuleList(AdaptiveHeadBlock(ch, i) for i in range(len(ch)))
        _make_head_end2end(self, end2end)

    def _adapt_features(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply adaptive attention to every output level."""
        return [adapter(x) for adapter in self.adapt]

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Adapt features, then run the standard Detect forward path."""
        return super().forward(self._adapt_features(x))
