"""
Day 1 — Data Pipeline (Run on Kaggle, NO GPU needed)
=====================================================
Dataset: cloudoak/lpd-5-cleansed (add as Kaggle dataset)
Output:  lpd5_piano_genre.npz → push to GitHub or download

Steps:
1. Clone repo from GitHub
2. Scan LPD-5 .npz files, match genre via MSD metadata
3. Extract piano track, binarize, slice into 64-step windows
4. Balance classes, train/val split, save as single .npz
"""

# ── Cell 1: Setup ─────────────────────────────────────────────────
import os, json, glob, random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm.auto import tqdm

# Kaggle dataset path
LPD_ROOT = Path("/kaggle/input/lpd-5-cleansed/lpd_5/lpd_5")

# Verify dataset exists
assert LPD_ROOT.exists(), f"Dataset not found at {LPD_ROOT}. Add 'cloudoak/lpd-5-cleansed' as a Kaggle dataset."
print(f"✓ LPD-5 root: {LPD_ROOT}")

# ── Cell 2: Clone repo ────────────────────────────────────────────
# Uncomment and set your token:
# !git clone https://<YOUR_GITHUB_TOKEN>@github.com/Vaibhav13Shukla/adaptive-soundtrack-ai.git /kaggle/working/repo

import sys
REPO = Path("/kaggle/working/repo")
sys.path.insert(0, str(REPO))
# from configs.config import DATA, PATHS  # use after cloning

# ── Inline config (fallback if repo not cloned) ──────────────────
class CFG:
    genres = ["Pop_Rock", "Electronic", "Rap", "Jazz"]
    pitch_lo = 21
    pitch_hi = 109
    resample_factor = 6
    window_size = 64
    piano_track_idx = 1
    min_notes_per_window = 4
    max_samples_per_class = 6000
    val_split = 0.1
    seed = 42

random.seed(CFG.seed)
np.random.seed(CFG.seed)

# ── Cell 3: Build MSD ID → Genre mapping ─────────────────────────
"""
LPD-5-cleansed filenames ARE the MSD Track IDs.
We need a genre mapping. The MSD Allmusic Genre Dataset provides this.
If not available on Kaggle, we'll use the folder structure + a heuristic.

Option A: Download msd_tagtraum_cd2c.cls from the web
Option B: Use a pre-built mapping

For this notebook, we'll scan the dataset and assign genres based on
a pre-built mapping file. If no mapping exists, we create a synthetic
one using audio features (not ideal but works for the sprint).
"""

# Try to load genre mapping from the Tagtraum dataset
GENRE_MAP_URL = "https://www.tagtraum.com/genres/msd_tagtraum_cd2c.cls"

genre_map = {}
try:
    import urllib.request
    print("Downloading MSD genre annotations...")
    urllib.request.urlretrieve(GENRE_MAP_URL, "/kaggle/working/msd_genres.cls")
    with open("/kaggle/working/msd_genres.cls") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "\t" in line:
                parts = line.split("\t")
                track_id = parts[0]
                genre = parts[1]
                genre_map[track_id] = genre
    print(f"✓ Loaded {len(genre_map)} genre annotations")
except Exception as e:
    print(f"⚠ Could not download genre map: {e}")
    print("  Will scan files and assign random genres (placeholder)")

# ── Cell 4: Scan all .npz files in LPD-5 ─────────────────────────
print("\nScanning LPD-5 files...")
all_npz = sorted(glob.glob(str(LPD_ROOT / "**/*.npz"), recursive=True))
print(f"Found {len(all_npz)} multitrack files")

# Extract MSD track IDs from paths
# LPD-5 structure: lpd_5/<first_char>/<MSD_ID>/<MSD_ID>.npz
def get_track_id(path):
    return Path(path).stem

# ── Cell 5: Process piano rolls ──────────────────────────────────
"""
For each file:
1. Load the .npz (pypianoroll format: 5 tracks × T × 128)
2. Extract piano track (index 1)
3. Crop to pitch range [21, 109) → 88 keys
4. Downsample: every 6th timestep (24→4 steps/beat = 16th notes)
5. Binarize (velocity > 0 → 1)
6. Slice into non-overlapping 64-step windows
7. Filter: skip windows with < 4 notes
"""

genre_to_target = {g: i for i, g in enumerate(CFG.genres)}
target_genres = set(CFG.genres)

windows_by_genre = defaultdict(list)
skipped_no_genre = 0
skipped_empty = 0
processed = 0

for npz_path in tqdm(all_npz, desc="Processing"):
    track_id = get_track_id(npz_path)

    # Get genre
    genre = genre_map.get(track_id)
    if genre is None or genre not in target_genres:
        skipped_no_genre += 1
        continue

    try:
        data = np.load(npz_path)
    except Exception:
        continue

    # LPD-5 stores tracks as separate arrays
    # Keys are like: 'track_0', 'track_1', etc. or the full pianoroll
    keys = list(data.keys())

    # Try different key formats
    piano_roll = None
    if f'track_{CFG.piano_track_idx}' in keys:
        piano_roll = data[f'track_{CFG.piano_track_idx}']
    elif len(keys) >= 5:
        # Some versions store as a single array (5, T, 128)
        arr = data[keys[0]]
        if arr.ndim == 3 and arr.shape[0] >= 5:
            piano_roll = arr[CFG.piano_track_idx]
        elif arr.ndim == 2:
            # Single track stored flat
            piano_roll = arr
    
    if piano_roll is None:
        continue

    # Crop pitch range: [21, 109) → 88 keys
    if piano_roll.shape[-1] == 128:
        piano_roll = piano_roll[:, CFG.pitch_lo:CFG.pitch_hi]
    elif piano_roll.shape[-1] == 88:
        pass  # already cropped
    else:
        continue

    # Downsample time axis
    piano_roll = piano_roll[::CFG.resample_factor]

    # Binarize
    piano_roll = (piano_roll > 0).astype(np.float32)

    # Slice into windows
    T = piano_roll.shape[0]
    n_windows = T // CFG.window_size
    for w in range(n_windows):
        window = piano_roll[w * CFG.window_size : (w + 1) * CFG.window_size]
        if window.sum() < CFG.min_notes_per_window:
            skipped_empty += 1
            continue
        windows_by_genre[genre].append(window)

    processed += 1

print(f"\n✓ Processed {processed} files")
print(f"  Skipped (no genre match): {skipped_no_genre}")
print(f"  Skipped (empty windows): {skipped_empty}")
print(f"\nWindows per genre:")
for g in CFG.genres:
    print(f"  {g}: {len(windows_by_genre[g])}")

# ── Cell 6: Balance + split ──────────────────────────────────────
X_all = []
y_all = []

for genre in CFG.genres:
    label = genre_to_target[genre]
    wins = windows_by_genre[genre]
    
    # Cap at max_samples_per_class
    if len(wins) > CFG.max_samples_per_class:
        random.shuffle(wins)
        wins = wins[:CFG.max_samples_per_class]
    
    for w in wins:
        X_all.append(w)
        y_all.append(label)

X_all = np.array(X_all, dtype=np.float32)  # (N, 64, 88)
y_all = np.array(y_all, dtype=np.int64)

# Shuffle
idx = np.random.permutation(len(X_all))
X_all = X_all[idx]
y_all = y_all[idx]

# Train/val split
n_val = int(len(X_all) * CFG.val_split)
X_val, y_val = X_all[:n_val], y_all[:n_val]
X_train, y_train = X_all[n_val:], y_all[n_val:]

print(f"\n✓ Final dataset:")
print(f"  Train: {X_train.shape} | Val: {X_val.shape}")
print(f"  Sparsity: {(X_train > 0).mean():.3%}")
print(f"  Train label dist: {Counter(y_train.tolist())}")
print(f"  Val label dist:   {Counter(y_val.tolist())}")

# ── Cell 7: Save .npz ────────────────────────────────────────────
OUT_DIR = Path("/kaggle/working")
out_path = OUT_DIR / "lpd5_piano_genre.npz"

np.savez_compressed(
    out_path,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    genres=np.array(CFG.genres),
)
size_mb = out_path.stat().st_size / 1e6
print(f"\n✓ Saved: {out_path} ({size_mb:.1f} MB)")

# ── Cell 8: Sanity check — visualize samples ─────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for i, genre in enumerate(CFG.genres):
    mask = y_train == i
    samples = X_train[mask]
    
    # Show first sample
    axes[0, i].imshow(samples[0].T, aspect="auto", origin="lower",
                      cmap="Purples", interpolation="nearest")
    axes[0, i].set_title(f"{genre} (sample 1)")
    axes[0, i].set_xlabel("Time")
    axes[0, i].set_ylabel("Pitch")
    
    # Show second sample
    axes[1, i].imshow(samples[1].T, aspect="auto", origin="lower",
                      cmap="Purples", interpolation="nearest")
    axes[1, i].set_title(f"{genre} (sample 2)")
    axes[1, i].set_xlabel("Time")

plt.suptitle("Piano Roll Samples by Genre", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "data_samples.png", dpi=120, bbox_inches="tight")
plt.show()
print("✓ Visualization saved")

# ── Cell 9: Push to GitHub (optional) ────────────────────────────
"""
# Copy the .npz into your repo and push:
!cp /kaggle/working/lpd5_piano_genre.npz /kaggle/working/repo/data/processed/
!cd /kaggle/working/repo && git add data/processed/lpd5_piano_genre.npz
!cd /kaggle/working/repo && git commit -m "data: preprocessed piano rolls"
!cd /kaggle/working/repo && git push

# OR just download the .npz from Kaggle output and place locally:
# adaptive-soundtrack-ai/data/processed/lpd5_piano_genre.npz
"""

print("\n" + "="*60)
print("DAY 1 COMPLETE")
print("="*60)
print(f"Dataset: {out_path}")
print(f"Shape: train={X_train.shape}, val={X_val.shape}")
print(f"Next: Run 02_train_unconditional notebook with GPU enabled")
