"""
Day 4 — Evaluation + Final Plots (Run locally or on Kaggle)
=============================================================
Prerequisites: checkpoint_best.pt in checkpoints/
Output: All plots for README, evaluation metrics table
"""

import sys, os, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from configs.config import MODEL, INFER, DATA, PATHS
from src.unet import UNet
from src.ddpm import DDPM
from src.evaluate import evaluate_batch
from src.midi_utils import piano_roll_to_midi

PATHS.make_dirs()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model ────────────────────────────────────────────────────
ckpt_path = PATHS.checkpoints / PATHS.best_ckpt
assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

unet = UNet(MODEL).to(device)
ddpm = DDPM(unet, MODEL).to(device)
state = torch.load(ckpt_path, map_location=device)
ddpm.load_state_dict(state["model"])
ddpm.eval()
print(f"✓ Loaded checkpoint (epoch {state['epoch']}, val_loss {state['val_loss']:.4f})")

# ── 1. Grid of samples per genre ─────────────────────────────────
N_PER_GENRE = 4
fig, axes = plt.subplots(len(DATA.genres), N_PER_GENRE, figsize=(4*N_PER_GENRE, 4*len(DATA.genres)))

for i, genre in enumerate(DATA.genres):
    g = torch.full((N_PER_GENRE,), i, device=device).long()
    shape = (N_PER_GENRE, 1, DATA.window_size, 88)
    x = ddpm.ddim_sample(shape, g, ddim_steps=50, cfg_scale=3.0)
    rolls = (x.squeeze(1).cpu().numpy() > 0).astype(np.float32)

    for j in range(N_PER_GENRE):
        axes[i, j].imshow(rolls[j].T, aspect="auto", origin="lower",
                          cmap="Purples", interpolation="nearest")
        axes[i, j].set_title(f"{genre} #{j+1}", fontsize=10)
        if j == 0:
            axes[i, j].set_ylabel("Pitch")
        axes[i, j].set_xlabel("Time")

        midi_path = PATHS.midi_examples / f"eval_{genre}_{j}.mid"
        piano_roll_to_midi(rolls[j], str(midi_path), pitch_lo=DATA.pitch_lo)

plt.suptitle("Generated Piano Rolls by Genre (CFG=3.0)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PATHS.plots / "genre_grid.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Genre grid saved")

# ── 2. CFG scale comparison ──────────────────────────────────────
scales = [0.0, 1.0, 3.0, 5.0, 7.0]
genre_idx = 3  # Jazz
fig, axes = plt.subplots(1, len(scales), figsize=(4*len(scales), 4))

for i, scale in enumerate(scales):
    g = torch.full((1,), genre_idx, device=device).long()
    shape = (1, 1, DATA.window_size, 88)
    x = ddpm.ddim_sample(shape, g, ddim_steps=50, cfg_scale=scale)
    roll = (x.squeeze().cpu().numpy() > 0).astype(np.float32)
    axes[i].imshow(roll.T, aspect="auto", origin="lower",
                   cmap="Purples", interpolation="nearest")
    axes[i].set_title(f"CFG={scale}")
    axes[i].set_xlabel("Time")

axes[0].set_ylabel("Pitch")
plt.suptitle(f"Effect of CFG Scale — {DATA.genres[genre_idx]}", fontweight="bold")
plt.tight_layout()
plt.savefig(PATHS.plots / "cfg_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 3. Metrics table ─────────────────────────────────────────────
N_EVAL = 16
results = {}

print(f"\n{'Genre':<15} {'Entropy':>8} {'Density':>8} {'Scale%':>8} {'Empty%':>8}")
print("=" * 55)

for i, genre in enumerate(DATA.genres):
    g = torch.full((N_EVAL,), i, device=device).long()
    shape = (N_EVAL, 1, DATA.window_size, 88)
    x = ddpm.ddim_sample(shape, g, ddim_steps=50, cfg_scale=3.0)
    rolls = (x.squeeze(1).cpu().numpy() > 0).astype(np.float32)
    m = evaluate_batch(rolls)
    results[genre] = m
    print(f"{genre:<15} {m['pitch_entropy']:8.2f} {m['note_density']:8.2f} "
          f"{m['scale_consistency']:8.1%} {m['empty_ratio']:8.1%}")

# Save metrics as JSON
with open(PATHS.logs / "eval_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Metrics saved to {PATHS.logs / 'eval_metrics.json'}")

# ── 4. Training curves (if log exists) ───────────────────────────
log_path = PATHS.logs / "train_log.json"
if log_path.exists():
    with open(log_path) as f:
        records = json.load(f)

    epochs = [r["epoch"] for r in records]
    train_l = [r["train_loss"] for r in records]
    val_l = [r["val_loss"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_l, label="Train", linewidth=2, color="#7C3AED")
    ax.plot(epochs, val_l, label="Val", linewidth=2, color="#F59E0B")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Training Curves"); ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PATHS.plots / "training_curves_final.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\n✓ All evaluation complete. Check plots/ and outputs/midi_examples/")
