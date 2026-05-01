"""
Conditional U-Net for DDPM on piano rolls.

Input:  (B, 1, 64, 88) noisy piano roll + timestep t + genre g
Output: (B, 1, 64, 88) predicted noise ε

WHY this design:
- Treat piano roll as 1-channel 2D image. Same U-Net pattern as image diffusion.
- Genre conditioning via learned embedding added to time embedding.
  Simple, sufficient for 4 classes. Class slide showed exactly this.
- FiLM conditioning in ResBlocks (scale + shift) — standard DDPM trick.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
sys.path.append(".")
from configs.config import MODEL


# ── Sinusoidal timestep embedding ─────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    """
    Same idea as transformer positional encoding.
    Encodes the diffusion timestep t into a smooth high-dim vector.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ── Time + genre embedding ────────────────────────────────────────
class TimeGenreEmbedding(nn.Module):
    """
    Time MLP + genre embedding. Outputs unified (B, time_dim).
    Genre vocabulary is n_genres + 1 — last index is the null/uncond token for CFG.
    """
    def __init__(self, base_ch: int, n_genres: int):
        super().__init__()
        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.genre_emb = nn.Embedding(n_genres + 1, time_dim)
        self.out_dim   = time_dim
        self.n_genres  = n_genres

    def forward(self, t: torch.Tensor, g: Optional[torch.Tensor] = None):
        emb = self.time_mlp(t)
        if g is not None:
            emb = emb + self.genre_emb(g)
        return emb


# ── Residual block with FiLM conditioning ─────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch),  in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.drop  = nn.Dropout(dropout)

        # Project time emb -> scale + shift (FiLM)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),
        )
        self.res_conv = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(temb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.drop(self.conv2(F.silu(h)))
        return h + self.res_conv(x)


# ── Down / Up sample ──────────────────────────────────────────────
class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ── Full U-Net ─────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    Encoder ↓ Bottleneck → Decoder ↑ with skip connections.
    Time + genre embedding is injected into every ResBlock.
    """
    def __init__(self, cfg=MODEL):
        super().__init__()
        ch    = cfg.base_channels
        mults = cfg.channel_mults
        chs   = [ch * m for m in mults]   # e.g. [64, 128, 256]

        self.cond = TimeGenreEmbedding(ch, cfg.n_genres)
        time_dim  = self.cond.out_dim

        self.init_conv = nn.Conv2d(cfg.in_channels, ch, 3, padding=1)

        # Encoder
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_ch = ch
        skip_chs = []
        for i, out_ch in enumerate(chs):
            blocks = nn.ModuleList([
                ResBlock(in_ch if j == 0 else out_ch, out_ch, time_dim, cfg.dropout)
                for j in range(cfg.n_res_blocks)
            ])
            self.enc.append(blocks)
            skip_chs.append(out_ch)
            self.downs.append(Downsample(out_ch) if i < len(chs) - 1 else nn.Identity())
            in_ch = out_ch

        # Bottleneck
        self.mid1 = ResBlock(in_ch, in_ch, time_dim, cfg.dropout)
        self.mid2 = ResBlock(in_ch, in_ch, time_dim, cfg.dropout)

        # Decoder
        self.dec = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, out_ch in enumerate(reversed(chs)):
            skip_ch = skip_chs[-(i + 1)]
            blocks = nn.ModuleList([
                ResBlock(in_ch + skip_ch if j == 0 else out_ch,
                         out_ch, time_dim, cfg.dropout)
                for j in range(cfg.n_res_blocks + 1)
            ])
            self.dec.append(blocks)
            self.ups.append(Upsample(out_ch) if i < len(chs) - 1 else nn.Identity())
            in_ch = out_ch

        # Output head
        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, cfg.in_channels, 1)

    def forward(self, x, t, g=None):
        temb  = self.cond(t, g)
        h     = self.init_conv(x)
        skips = []

        # Encoder
        for blocks, ds in zip(self.enc, self.downs):
            for blk in blocks:
                h = blk(h, temb)
            skips.append(h)
            h = ds(h)

        # Bottleneck
        h = self.mid1(h, temb)
        h = self.mid2(h, temb)

        # Decoder
        for blocks, us in zip(self.dec, self.ups):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, temb)
            h = us(h)

        return self.out_conv(F.silu(self.out_norm(h)))
