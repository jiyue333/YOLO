# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "CoordAtt",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "DySample",
    "Focus",
    "FreqFusion",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class CoordAtt(nn.Module):
    """Coordinate Attention for small-object neck features.

    References:
        https://arxiv.org/abs/2103.02907
        https://github.com/houqb/CoordAttention
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 32):
        """Initialize Coordinate Attention with optional channel projection."""
        super().__init__()
        c2 = c1 if c2 is None else c2
        mip = max(8, c1 // reduction)
        self.proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1)
        self.conv1 = nn.Conv2d(c2, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply coordinate attention over height and width separately."""
        x = self.proj(x)
        identity = x
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat((x_h, x_w), dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return identity * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


class DySample(nn.Module):
    """Dynamic upsampling via learned offsets and ``grid_sample``.

    This follows the dependency-free ``lp`` path from the official DySample implementation and keeps an optional
    projection so YOLO YAML can request a target channel count.

    References:
        https://arxiv.org/abs/2308.15085
        https://github.com/tiny-smart/dysample
    """

    def __init__(self, c1: int, c2: int | None = None, scale: int = 2, groups: int = 4, dyscope: bool = False):
        """Initialize an offset-based DySample upsampler."""
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.scale = scale
        self.proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1)
        self.groups = max(math.gcd(c2, groups), 1)
        offset_channels = 2 * self.groups * scale**2
        self.offset = nn.Conv2d(c2, offset_channels, 1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.offset.bias, 0.0)
        if dyscope:
            self.scope = nn.Conv2d(c2, offset_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0.0)
        self.register_buffer("init_pos", self._init_pos(), persistent=False)

    def _init_pos(self) -> torch.Tensor:
        """Return the base sub-pixel offsets used by DySample."""
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        grid = torch.stack(torch.meshgrid(h, h, indexing="ij")).transpose(1, 2)
        return grid.repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def _sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Sample ``x`` at learned sub-pixel coordinates."""
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij")).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
            .view(b, 2, -1, self.scale * h, self.scale * w)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(b, -1, self.scale * h, self.scale * w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample feature maps with learned offsets."""
        x = self.proj(x)
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self._sample(x, offset)


class FreqFusion(nn.Module):
    """Frequency-aware fusion of high-resolution and low-resolution features.

    The module mirrors the official operator shape: HR/LR channel compressors generate adaptive low-pass and high-pass
    masks, the LR feature is upsampled by CARAFE-style content-aware reassembly, and the HR feature contributes a
    high-frequency residual. The implementation is pure PyTorch so it can run without MMCV custom ops.

    References:
        https://arxiv.org/abs/2408.12879
        https://github.com/Linwei-Chen/FreqFusion
    """

    def __init__(
        self,
        lr_channels: int,
        hr_channels: int | None = None,
        c2: int | None = None,
        scale: int = 2,
        lowpass_kernel: int = 5,
        highpass_kernel: int = 3,
        compressed_channels: int = 64,
        up_group: int = 1,
        hr_residual: bool = True,
    ):
        """Initialize adaptive low/high-pass fusion for an FPN upsample step."""
        super().__init__()
        hr_channels = lr_channels if hr_channels is None else hr_channels
        c2 = lr_channels if c2 is None else c2
        self.scale = scale
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.hr_residual = hr_residual
        compressed_channels = min(compressed_channels, max(lr_channels, hr_channels))
        self.hr_channel_compressor = nn.Conv2d(hr_channels, compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, compressed_channels, 1)
        self.content_encoder = nn.Conv2d(compressed_channels, lowpass_kernel**2 * up_group, 3, padding=1)
        self.content_encoder2 = nn.Conv2d(compressed_channels, highpass_kernel**2 * up_group, 3, padding=1)
        self.lr_out = Conv(lr_channels, c2, 1)
        self.hr_out = Conv(hr_channels, c2, 1)
        self.register_buffer("hamming_lowpass", self._hamming2d(lowpass_kernel), persistent=False)
        self.register_buffer("hamming_highpass", self._hamming2d(highpass_kernel), persistent=False)
        self._init_weights()

    @staticmethod
    def _hamming2d(kernel: int) -> torch.Tensor:
        """Create the 2D Hamming window used to regularize generated kernels."""
        window = np.outer(np.hamming(kernel), np.hamming(kernel)).astype(np.float32)
        return torch.from_numpy(window)[None, None]

    def _init_weights(self) -> None:
        """Match FreqFusion's small-random mask generator initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.normal_(self.content_encoder.weight, mean=0.0, std=0.001)
        nn.init.normal_(self.content_encoder2.weight, mean=0.0, std=0.001)

    def _kernel_normalizer(self, mask: torch.Tensor, kernel: int, hamming: torch.Tensor) -> torch.Tensor:
        """Normalize generated kernels over the spatial support, following the official FreqFusion code."""
        n, mask_c, h, w = mask.shape
        mask_channel = mask_c // kernel**2
        mask = mask.view(n, mask_channel, kernel**2, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).reshape(n, -1, kernel, kernel)
        mask = mask * hamming.to(device=mask.device, dtype=mask.dtype)
        mask = mask / mask.sum(dim=(-1, -2), keepdim=True).clamp_min(torch.finfo(mask.dtype).eps)
        return mask.view(n, mask_channel, h, w, -1).permute(0, 1, 4, 2, 3).reshape(n, -1, h, w)

    @staticmethod
    def _carafe(x: torch.Tensor, normed_mask: torch.Tensor, kernel: int, group: int = 1, up: int = 1) -> torch.Tensor:
        """Content-aware feature reassembly fallback equivalent to the official non-MMCV path."""
        b, c, h, w = x.shape
        _, _, mh, mw = normed_mask.shape
        pad = kernel // 2
        pad_mode = "reflect" if h > pad and w > pad else "replicate"
        unfold_x = F.unfold(F.pad(x, [pad] * 4, mode=pad_mode), kernel_size=(kernel, kernel))
        unfold_x = unfold_x.view(b, c * kernel * kernel, h, w)
        target_size = (mh, mw)
        if up != 1 or (h, w) != target_size:
            unfold_x = F.interpolate(unfold_x, size=target_size, mode="nearest")
        unfold_x = unfold_x.view(b, c, kernel * kernel, mh, mw)
        if group <= 1:
            mask = normed_mask.view(b, 1, kernel * kernel, mh, mw)
            return (unfold_x * mask).sum(dim=2)
        if c % group != 0:
            raise ValueError(f"FreqFusion CARAFE group={group} must divide channels={c}.")
        mask = normed_mask.view(b, group, kernel * kernel, mh, mw)
        unfold_x = unfold_x.view(b, group, c // group, kernel * kernel, mh, mw)
        return (unfold_x * mask.unsqueeze(2)).sum(dim=3).view(b, c, mh, mw)

    def forward(self, x: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse ``[low_resolution, high_resolution]`` features and return the upsampled LR path."""
        if isinstance(x, (list, tuple)):
            lr_feat, hr_feat = x
        else:
            lr_feat = x
            hr_feat = F.interpolate(x, scale_factor=self.scale, mode="nearest")

        compressed_hr = self.hr_channel_compressor(hr_feat)
        compressed_lr = self.lr_channel_compressor(lr_feat)

        mask_hr = self._kernel_normalizer(
            self.content_encoder2(compressed_hr), self.highpass_kernel, self.hamming_highpass
        )
        compressed_hr = compressed_hr + compressed_hr - self._carafe(
            compressed_hr, mask_hr, self.highpass_kernel, self.up_group, 1
        )

        mask_lr_hr = self.content_encoder(compressed_hr)
        mask_lr_init = self._kernel_normalizer(mask_lr_hr, self.lowpass_kernel, self.hamming_lowpass)
        mask_lr_from_lr = self._carafe(
            self.content_encoder(compressed_lr), mask_lr_init, self.lowpass_kernel, self.up_group, self.scale
        )
        if mask_lr_from_lr.shape[-2:] != compressed_hr.shape[-2:]:
            mask_lr_from_lr = F.interpolate(mask_lr_from_lr, size=compressed_hr.shape[-2:], mode="nearest")
        mask_lr = self._kernel_normalizer(mask_lr_hr + mask_lr_from_lr, self.lowpass_kernel, self.hamming_lowpass)

        lr_feat = self._carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, self.scale)
        if lr_feat.shape[-2:] != hr_feat.shape[-2:]:
            lr_feat = F.interpolate(lr_feat, size=hr_feat.shape[-2:], mode="bilinear", align_corners=False)

        hr_high = hr_feat - self._carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)
        hr_feat = hr_feat + hr_high if self.hr_residual else hr_high
        return self.lr_out(lr_feat) + self.hr_out(hr_feat)


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply convolution transpose and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
