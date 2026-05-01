"""
Training loop with auto-resume and periodic checkpointing.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import time

import sys
sys.path.append(".")
from configs.config import TRAIN, INFER, PATHS
from src.logger import ExperimentLogger


class Trainer:
    def __init__(self, ddpm: nn.Module, optimizer, device: str,
                 ckpt_dir: Path, log_path: Path):
        self.ddpm       = ddpm
        self.optimizer  = optimizer
        self.device     = device
        self.ckpt_dir   = Path(ckpt_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger     = ExperimentLogger(log_path)
        self.start_epoch = 0
        self.best_val    = float("inf")

    # ── Resume logic ──────────────────────────────────────────────
    def maybe_resume(self):
        latest = self.ckpt_dir / PATHS.latest_ckpt
        if not latest.exists() or not TRAIN.resume:
            print("Starting from scratch")
            return
        ckpt = torch.load(latest, map_location=self.device)
        self.ddpm.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val    = ckpt.get("best_val", float("inf"))
        print(f"✓ Resumed from epoch {self.start_epoch}, best_val={self.best_val:.4f}")

    # ── Train one epoch ───────────────────────────────────────────
    def train_epoch(self, loader: DataLoader) -> float:
        self.ddpm.train()
        total, n = 0.0, 0
        pbar = tqdm(loader, leave=False)
        for x, g in pbar:
            x, g = x.to(self.device), g.to(self.device)
            self.optimizer.zero_grad()
            loss = self.ddpm.p_losses(x, g)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ddpm.parameters(), TRAIN.grad_clip)
            self.optimizer.step()
            total += loss.item() * x.size(0); n += x.size(0)
            pbar.set_description(f"loss {loss.item():.4f}")
        return total / n

    # ── Validate ──────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.ddpm.eval()
        total, n = 0.0, 0
        for x, g in loader:
            x, g = x.to(self.device), g.to(self.device)
            loss = self.ddpm.p_losses(x, g)
            total += loss.item() * x.size(0); n += x.size(0)
        return total / n

    # ── Save ──────────────────────────────────────────────────────
    def save(self, epoch: int, val_loss: float, is_best: bool = False):
        state = {
            "epoch":      epoch,
            "model":      self.ddpm.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "val_loss":   val_loss,
            "best_val":   self.best_val,
        }
        torch.save(state, self.ckpt_dir / PATHS.latest_ckpt)
        if is_best:
            torch.save(state, self.ckpt_dir / PATHS.best_ckpt)
            print(f"  ✓ NEW BEST: {val_loss:.4f}")

    # ── Full loop ─────────────────────────────────────────────────
    def fit(self, train_loader, val_loader, n_epochs: int):
        self.maybe_resume()
        for epoch in range(self.start_epoch, n_epochs):
            t0       = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.validate(val_loader)
            elapsed    = time.time() - t0

            print(f"Epoch {epoch:3d} | train {train_loss:.4f} | "
                  f"val {val_loss:.4f} | {elapsed:.1f}s")

            self.logger.log(
                epoch=epoch, train_loss=train_loss,
                val_loss=val_loss, elapsed_sec=elapsed,
            )

            is_best = val_loss < self.best_val
            if is_best:
                self.best_val = val_loss

            if epoch % TRAIN.save_every == 0 or is_best:
                self.save(epoch, val_loss, is_best=is_best)

        # Final save
        self.save(n_epochs - 1, val_loss, is_best=False)
        print("✓ Training complete")
