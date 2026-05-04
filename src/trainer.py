"""
Training loop with auto-resume and periodic checkpointing.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import time

from configs.config import TRAIN, PATHS
from src.logger import ExperimentLogger


class Trainer:
    def __init__(self, ddpm: nn.Module, optimizer, device: str,
                 ckpt_dir: Path, log_path: Path,
                 scheduler=None, use_amp: bool = False):
        self.ddpm       = ddpm
        self.optimizer  = optimizer
        self.device     = device
        self.scheduler  = scheduler
        self.ckpt_dir   = Path(ckpt_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger     = ExperimentLogger(log_path)
        self.start_epoch = 0
        self.best_val    = float("inf")
        # AMP scaler — no-op on CPU, active on CUDA when requested
        self.use_amp = use_amp and (device == "cuda")
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    # ── Resume logic ──────────────────────────────────────────────
    def maybe_resume(self):
        latest = self.ckpt_dir / PATHS.latest_ckpt
        if not latest.exists() or not TRAIN.resume:
            print("Starting from scratch")
            return
        ckpt = torch.load(latest, map_location=self.device, weights_only=True)
        self.ddpm.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scaler is not None and ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.scheduler is not None and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
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
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss = self.ddpm.p_losses(x, g)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.ddpm.parameters(), TRAIN.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item() * x.size(0); n += x.size(0)
            pbar.set_description(f"loss {loss.item():.4f}")
        return total / n

    # ── Validate ──────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.ddpm.eval()
        total, n = 0.0, 0
        # Fix RNG state so validation loss is reproducible across epochs,
        # making best-checkpoint selection meaningful.
        cpu_state   = torch.get_rng_state()
        cuda_states = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        for x, g in loader:
            x, g = x.to(self.device), g.to(self.device)
            loss = self.ddpm.p_losses(x, g)
            total += loss.item() * x.size(0); n += x.size(0)
        # Restore training RNG so training randomness is unaffected
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        return total / n

    # ── Save ──────────────────────────────────────────────────────
    def save(self, epoch: int, val_loss: float, is_best: bool = False):
        state = {
            "epoch":      epoch,
            "model":      self.ddpm.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scaler":     self.scaler.state_dict(),
            "scheduler":  self.scheduler.state_dict() if self.scheduler else None,
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
        if self.start_epoch >= n_epochs:
            print(f"Training already complete ({self.start_epoch} epochs done, target {n_epochs}).")
            return
        val_loss = float("inf")
        for epoch in range(self.start_epoch, n_epochs):
            t0         = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.validate(val_loader)
            elapsed    = time.time() - t0

            if self.scheduler is not None:
                self.scheduler.step()

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
