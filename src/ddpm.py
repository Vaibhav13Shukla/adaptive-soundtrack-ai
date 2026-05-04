"""
DDPM forward + reverse diffusion engine with classifier-free guidance.

Connects to course concepts:
- Forward q(x_t|x_0): direct noise sampling via reparameterization
- Reverse: U-Net learns to predict ε given x_t, t, g
- Score function: ∇_x log q(x_t) ≈ -ε_θ / sqrt(1 - ᾱ_t)
- DDIM: deterministic skip-step sampling (50 steps vs 1000)
- CFG: train both cond + uncond in one model, interpolate at inference
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from configs.config import MODEL


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, cfg=MODEL):
        super().__init__()
        self.model       = model
        self.T           = cfg.T
        self.cfg_dropout = cfg.cfg_dropout
        self.n_genres    = cfg.n_genres

        # Build noise schedule
        if cfg.beta_schedule == "cosine":
            betas = self._cosine_betas(cfg.T)
        else:
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)

        alphas     = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas",                       betas)
        self.register_buffer("alphas",                      alphas)
        self.register_buffer("alpha_bars",                  alpha_bars)
        self.register_buffer("sqrt_alpha_bars",             torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars",   torch.sqrt(1.0 - alpha_bars))

    @staticmethod
    def _cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
        """Improved cosine noise schedule (Nichol & Dhariwal, 2021).

        Distributes learning signal more evenly across timesteps than the
        linear schedule, particularly important for sparse data like piano rolls.
        """
        steps     = torch.arange(T + 1, dtype=torch.float64)
        f         = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas     = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(min=1e-5, max=0.9999).float()

    # ── Forward diffusion (training) ──────────────────────────────
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε  (reparameterization trick)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab  = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_1ab = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_1ab * noise, noise

    # ── Training loss ─────────────────────────────────────────────
    def p_losses(self, x0: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        DDPM loss = MSE(true_noise, predicted_noise)

        CFG trick: randomly null-out genre with prob cfg_dropout.
        This trains BOTH conditional and unconditional models
        in a single network. At inference, interpolate the two.
        """
        B      = x0.shape[0]
        device = x0.device

        t           = torch.randint(0, self.T, (B,), device=device).long()
        x_t, noise  = self.q_sample(x0, t)

        # Random null-out for CFG
        null_idx   = self.n_genres
        mask       = torch.rand(B, device=device) < self.cfg_dropout
        g_in       = g.clone()
        g_in[mask] = null_idx

        eps_pred = self.model(x_t, t, g_in)
        return F.mse_loss(eps_pred, noise)

    # ── DDIM sampling (inference) ─────────────────────────────────
    @torch.no_grad()
    def ddim_sample(self, shape: Tuple, g: torch.Tensor,
                    ddim_steps: int = 50, cfg_scale: float = 3.0,
                    eta: float = 0.0) -> torch.Tensor:
        """
        Fast deterministic sampling.
        ε_guided = ε_uncond + scale × (ε_cond - ε_uncond)
        Returns piano roll in [-1, 1] range — caller binarizes if needed.
        """
        device = next(self.model.parameters()).device
        # Normalize g to the model's device regardless of where the caller created it
        g      = g.to(device)
        B      = shape[0]
        x      = torch.randn(shape, device=device)
        g_null = torch.full((B,), self.n_genres, device=device).long()

        # Pick timesteps to visit (subsampled)
        times = torch.linspace(self.T - 1, 0, ddim_steps + 1).long().to(device)

        for i in range(len(times) - 1):
            t_cur, t_next = times[i].item(), times[i + 1].item()
            t_batch = torch.full((B,), t_cur, device=device).long()

            ab_cur  = self.alpha_bars[t_cur]
            ab_next = self.alpha_bars[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

            # CFG: predict twice, interpolate
            eps_cond   = self.model(x, t_batch, g)
            eps_uncond = self.model(x, t_batch, g_null)
            eps        = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            # DDIM step
            x0_pred = (x - torch.sqrt(1 - ab_cur) * eps) / torch.sqrt(ab_cur)
            x0_pred = x0_pred.clamp(-1, 1)

            sigma = (eta * torch.sqrt((1 - ab_next) / (1 - ab_cur))
                     * torch.sqrt(1 - ab_cur / ab_next))
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

            # Clamp before sqrt to prevent NaN when eta > 0 causes (1-ab_next-σ²) < 0
            dir_coef = (1 - ab_next - sigma ** 2).clamp(min=0.0)
            x = (torch.sqrt(ab_next) * x0_pred
                 + torch.sqrt(dir_coef) * eps
                 + sigma * noise)

        return x   # [-1, 1] range; caller does (x > 0) for binary
