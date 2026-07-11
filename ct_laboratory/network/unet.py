"""Configurable 2D/3D U-Net.

A dimension-agnostic U-Net built on plain ``nn.Conv2d`` / ``nn.Conv3d`` (no
``diffusers`` dependency). Switch between 2D and 3D with ``dim=2`` / ``dim=3``;
depth, width, and residual-block count are all configurable. An optional
sinusoidal timestep embedding makes it usable as a noise-prediction backbone for
diffusion models.

Added during the gmi -> ct_laboratory merge (2026-07-11). The gmi ``network``
area only shipped diffusers-based *2D* U-Nets; this provides a native,
diffusers-free U-Net that works in both 2D and 3D.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(dim):
    return nn.Conv2d if dim == 2 else nn.Conv3d


def _conv_transpose(dim):
    return nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d


def _num_groups(num_groups, channels):
    """Largest divisor of ``channels`` that is <= ``num_groups`` (GroupNorm needs
    ``channels % groups == 0``)."""
    g = min(num_groups, channels)
    while channels % g != 0:
        g -= 1
    return g


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        t = t.float().view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    """(GroupNorm -> SiLU -> Conv) x2 with a residual connection and optional
    additive timestep embedding."""

    def __init__(self, dim, in_channels, out_channels, num_groups, time_emb_dim=None):
        super().__init__()
        Conv = _conv(dim)
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(_num_groups(num_groups, in_channels), in_channels)
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(num_groups, out_channels), out_channels)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None
        self.skip = (
            Conv(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, temb=None):
        h = self.conv1(self.act(self.norm1(x)))
        if self.time_mlp is not None and temb is not None:
            bshape = (h.shape[0], -1) + (1,) * (h.dim() - 2)
            h = h + self.time_mlp(self.act(temb)).view(*bshape)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class ConfigurableUNet(nn.Module):
    """A configurable U-Net for 2D or 3D data.

    Args:
        dim: 2 or 3 (spatial dimensionality).
        in_channels: number of input channels.
        out_channels: number of output channels.
        base_channels: channel width of the first level.
        channel_mults: per-level channel multipliers; ``len`` sets the depth.
        num_res_blocks: residual blocks per level.
        norm_num_groups: max GroupNorm groups (clamped to divide the channel count).
        time_embedding: if True, accept a timestep ``t`` in ``forward`` and inject
            a sinusoidal embedding into every residual block.

    Note: spatial dimensions must be divisible by ``2 ** (len(channel_mults) - 1)``.
    """

    def __init__(
        self,
        dim=2,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        norm_num_groups=8,
        time_embedding=False,
    ):
        super().__init__()
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        self.dim = dim
        self.time_embedding = time_embedding

        Conv = _conv(dim)
        ConvT = _conv_transpose(dim)

        time_emb_dim = base_channels * 4 if time_embedding else None
        self.time_emb_dim = time_emb_dim
        if time_embedding:
            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(base_channels),
                nn.Linear(base_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.time_embed = None

        chs = [base_channels * m for m in channel_mults]
        levels = len(chs)
        self.init_conv = Conv(in_channels, chs[0], kernel_size=3, padding=1)

        # Encoder: one stack of residual blocks per level.
        self.enc = nn.ModuleList()
        prev = chs[0]
        for ch in chs:
            blocks = nn.ModuleList(
                [
                    ResBlock(dim, prev if j == 0 else ch, ch, norm_num_groups, time_emb_dim)
                    for j in range(num_res_blocks)
                ]
            )
            self.enc.append(blocks)
            prev = ch
        # Downsamplers between levels (levels-1 of them), strided conv.
        self.downs = nn.ModuleList(
            [Conv(chs[i], chs[i], kernel_size=3, stride=2, padding=1) for i in range(levels - 1)]
        )
        # Upsamplers (levels-1), transpose conv doubling spatial size.
        self.ups = nn.ModuleList(
            [ConvT(chs[i + 1], chs[i], kernel_size=4, stride=2, padding=1) for i in range(levels - 1)]
        )
        # Decoder residual stacks (levels-1): input is cat(up, skip) = 2*chs[i].
        self.dec = nn.ModuleList()
        for i in range(levels - 1):
            blocks = nn.ModuleList()
            in_ch = 2 * chs[i]
            for j in range(num_res_blocks):
                blocks.append(ResBlock(dim, in_ch if j == 0 else chs[i], chs[i], norm_num_groups, time_emb_dim))
            self.dec.append(blocks)

        self.out_norm = nn.GroupNorm(_num_groups(norm_num_groups, chs[0]), chs[0])
        self.out_act = nn.SiLU()
        self.out_conv = Conv(chs[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, t=None):
        temb = None
        if self.time_embedding:
            if t is None:
                t = torch.zeros(x.shape[0], device=x.device)
            temb = self.time_embed(t)

        h = self.init_conv(x)
        levels = len(self.enc)
        skips = []
        for i in range(levels):
            for block in self.enc[i]:
                h = block(h, temb)
            if i < levels - 1:
                skips.append(h)
                h = self.downs[i](h)

        for i in reversed(range(levels - 1)):
            h = self.ups[i](h)
            h = torch.cat([h, skips.pop()], dim=1)
            for block in self.dec[i]:
                h = block(h, temb)

        return self.out_conv(self.out_act(self.out_norm(h)))
