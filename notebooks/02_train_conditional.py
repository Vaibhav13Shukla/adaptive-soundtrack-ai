"""
Day 2 — Train Conditional DDPM (Run on Kaggle, GPU P100)
=========================================================
Prerequisites: 01_data_pipeline.py output (lpd5_piano_genre.npz)
Output: checkpoints/checkpoint_best.pt, training plots

Setup on Kaggle:
1. Enable GPU (P100)
2. Enable Internet
3. Add dataset: your previous notebook output OR upload the .npz
4. Clone your repo (has all src/ code)
"""

# ── Cell 1: Clone repo + install deps ────────────────────────────
"""
!git clone https://<TOKEN>@github.com/Vaibhav13Shukla/adaptive-soundtrack-ai.git /kaggle/working/repo
!pip install pretty_midi -q
"""

# ── Cell 2: Setup paths ──────────────────────────────────────────
import sys, os, shutil
from pathlib import Path

REPO = Path("/kaggle/working/repo")
sys.path.insert(0, str(REPO))

# Copy dataset into repo structure
DATA_SRC = Path("/kaggle/input")  # adjust to your dataset source
DEST = REPO / "data" / "processed"
DEST.mkdir(parents=True, exist_ok=True)

# Find the .npz — could be in notebook output or uploaded dataset
for candidate in [
    DATA_SRC / "lpd5-piano-genre" / "lpd5_piano_genre.npz",
    DATA_SRC / "lpd5_piano_genre.npz",
    Path("/kaggle/working/lpd5_piano_genre.npz"),
]:
    if candidate.exists():
        shutil.copy(candidate, DEST / "lpd5_piano_genre.npz")
        print(f"✓ Copied dataset from {candidate}")
        break
else:
    print("⚠ Dataset not found. Upload lpd5_piano_genre.npz as a Kaggle dataset.")

# ── Cell 3: Verify setup ─────────────────────────────────────────
import torch
import numpy as np
from configs.config import MODEL, TRAIN, PATHS, DATA

os.chdir(REPO)
PATHS.make_dirs()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

npz_path = PATHS.data_processed / PATHS.dataset_file
assert npz_path.exists(), f"Missing: {npz_path}"
print(f"✓ Dataset: {npz_path}")

# ── Cell 4: Load data ────────────────────────────────────────────
from src.dataset import make_dataloaders

train_loader, val_loader, genres = make_dataloaders(str(npz_path))

# Quick sanity check
x_batch, g_batch = next(iter(train_loader))
print(f"Batch shape: {x_batch.shape}")  # (64, 1, 64, 88)
print(f"Labels: {g_batch[:8]}")
print(f"Value range: [{x_batch.min():.1f}, {x_batch.max():.1f}]")

# ── Cell 5: Build model ──────────────────────────────────────────
from src.unet import UNet
from src.ddpm import DDPM

unet = UNet(MODEL).to(device)
ddpm = DDPM(unet, MODEL).to(device)
optimizer = torch.optim.AdamW(unet.parameters(), lr=TRAIN.lr)

n_params = sum(p.numel() for p in unet.parameters())
print(f"✓ Model: {n_params:,} parameters")

# Verify forward pass
with torch.no_grad():
    x_test = x_batch[:2].to(device)
    g_test = g_batch[:2].to(device)
    loss = ddpm.p_losses(x_test, g_test)
    print(f"  Test loss: {loss.item():.4f}")

# ── Cell 6: Train ────────────────────────────────────────────────
from src.trainer import Trainer

trainer = Trainer(
    ddpm, optimizer, device,
    ckpt_dir=PATHS.checkpoints,
    log_path=PATHS.logs / "train_log.json",
)

# This will auto-resume if checkpoint exists
trainer.fit(train_loader, val_loader, n_epochs=TRAIN.n_epochs)

# ── Cell 7: Plot training curves ─────────────────────────────────
import matplotlib.pyplot as plt
import json

log_path = PATHS.logs / "train_log.json"
with open(log_path) as f:
    records = json.load(f)

epochs = [r["epoch"] for r in records]
train_losses = [r["train_loss"] for r in records]
val_losses = [r["val_loss"] for r in records]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_losses, label="Train", linewidth=2)
ax1.plot(epochs, val_losses, label="Val", linewidth=2)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss")
ax1.set_title("Training Curves"); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(epochs[-20:], train_losses[-20:], label="Train", linewidth=2)
ax2.plot(epochs[-20:], val_losses[-20:], label="Val", linewidth=2)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE Loss")
ax2.set_title("Last 20 Epochs (zoomed)"); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PATHS.plots / "training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Cell 8: Generate samples ─────────────────────────────────────
from src.midi_utils import piano_roll_to_midi

ddpm.eval()
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, genre in enumerate(DATA.genres):
    g = torch.full((1,), i, device=device).long()
    shape = (1, 1, DATA.window_size, 88)
    x = ddpm.ddim_sample(shape, g, ddim_steps=50, cfg_scale=3.0)
    roll = (x.squeeze().cpu().numpy() > 0).astype(np.float32)
    
    axes[i].imshow(roll.T, aspect="auto", origin="lower",
                   cmap="Purples", interpolation="nearest")
    axes[i].set_title(genre)
    axes[i].set_xlabel("Time")
    
    midi_path = PATHS.midi_examples / f"sample_{genre}.mid"
    piano_roll_to_midi(roll, str(midi_path), pitch_lo=DATA.pitch_lo)

plt.suptitle("Generated Samples (CFG=3.0, DDIM 50 steps)", fontweight="bold")
plt.tight_layout()
plt.savefig(PATHS.plots / "generated_samples.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Samples generated + saved as MIDI")

# ── Cell 9: Evaluate ─────────────────────────────────────────────
from src.evaluate import evaluate_batch

# Generate a batch per genre for metrics
print("\nMetrics per genre:")
print(f"{'Genre':<15} {'Entropy':>8} {'Density':>8} {'Scale%':>8} {'Empty%':>8}")
print("-" * 55)

for i, genre in enumerate(DATA.genres):
    g = torch.full((8,), i, device=device).long()
    shape = (8, 1, DATA.window_size, 88)
    x = ddpm.ddim_sample(shape, g, ddim_steps=50, cfg_scale=3.0)
    rolls = (x.squeeze(1).cpu().numpy() > 0).astype(np.float32)
    m = evaluate_batch(rolls)
    print(f"{genre:<15} {m['pitch_entropy']:8.2f} {m['note_density']:8.2f} "
          f"{m['scale_consistency']:8.1%} {m['empty_ratio']:8.1%}")

# ── Cell 10: Push checkpoint to GitHub ────────────────────────────
"""
!cd /kaggle/working/repo && git add checkpoints/ plots/ logs/ outputs/
!cd /kaggle/working/repo && git commit -m "train: conditional DDPM checkpoint + plots"
!cd /kaggle/working/repo && git push
"""

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print("Next steps:")
print("1. Download checkpoint_best.pt from checkpoints/")
print("2. Place in your local repo: checkpoints/checkpoint_best.pt")
print("3. Run: streamlit run app.py")
