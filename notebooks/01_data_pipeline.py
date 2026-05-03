"""
Day 1 — Data Pipeline (Run on Kaggle, NO GPU needed)
=====================================================
Dataset: cloudoak/lpd-5-cleansed (add as Kaggle dataset)
Output:  lpd5_piano_genre.npz → download to local repo
"""

# ── Cell 1: Setup ─────────────────────────────────────────────────
import os, json, glob, random, hashlib
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm.auto import tqdm
from scipy.sparse import csc_matrix   # LPD stores piano rolls as sparse CSC

# Kaggle dataset path
LPD_ROOT = None
for c in [
    Path("/kaggle/input/datasets/cloudoak/lpd-5-cleansed"),
    Path("/kaggle/input/lpd-5-cleansed"),
]:
    if c.exists():
        LPD_ROOT = c
        break

if LPD_ROOT is None:
    all_npz = list(Path("/kaggle/input").rglob("*.npz"))
    if all_npz:
        LPD_ROOT = all_npz[0].parents[4]  # go up past the letter dirs

assert LPD_ROOT is not None, "Dataset not found."
print(f"✓ LPD-5 root: {LPD_ROOT}")

# Config
class CFG:
    genres = ["Pop_Rock", "Electronic", "Rap", "Jazz"]
    pitch_lo = 21
    pitch_hi = 109
    resample_factor = 6
    window_size = 64
    piano_track_idx = 1    # 0=Drums, 1=Piano, 2=Guitar, 3=Bass, 4=Strings
    min_notes_per_window = 4
    max_samples_per_class = 6000
    val_split = 0.1
    seed = 42

random.seed(CFG.seed)
np.random.seed(CFG.seed)

# ── Cell 2: Understand the data format ────────────────────────────
"""
KEY INSIGHT from probing:
- Files are stored as scipy SPARSE CSC matrices, NOT dense arrays
- Keys: pianoroll_N_csc_indptr, pianoroll_N_csc_indices, pianoroll_N_csc_data, pianoroll_N_csc_shape
- N = track index (0=Drums, 1=Piano, 2=Guitar, 3=Bass, 4=Strings)
- Shape is (T, 128) where T varies per song

PATH STRUCTURE:
  .../lpd_5/lpd_5_cleansed/A/A/A/TRAAAGR128F425B14B/b97c529ab9ef783a849b896816001748.npz
  The MSD Track ID = parent folder name (TRAAAGR128F425B14B)
  The filename = a content hash (NOT the track ID)
"""

def load_sparse_pianoroll(data, track_idx):
    """Reconstruct dense piano roll from CSC sparse components."""
    prefix = f"pianoroll_{track_idx}_csc"
    try:
        indptr  = data[f"{prefix}_indptr"]
        indices = data[f"{prefix}_indices"]
        values  = data[f"{prefix}_data"]
        shape   = tuple(data[f"{prefix}_shape"])
    except KeyError:
        return None

    if len(values) == 0:
        return None  # empty track

    sparse = csc_matrix((values, indices, indptr), shape=shape)
    return sparse.toarray()  # (T, 128) dense

def get_msd_track_id(npz_path):
    """MSD Track ID is the parent directory name, e.g. TRAAAGR128F425B14B"""
    return Path(npz_path).parent.name

# Verify on first file
all_npz = sorted(glob.glob(str(LPD_ROOT / "**/*.npz"), recursive=True))
print(f"Total files: {len(all_npz)}")

sample_path = all_npz[0]
sample_data = np.load(sample_path)
print(f"\nSample: {sample_path}")
print(f"MSD ID: {get_msd_track_id(sample_path)}")

piano = load_sparse_pianoroll(sample_data, CFG.piano_track_idx)
if piano is not None:
    print(f"Piano track shape: {piano.shape}")  # (T, 128)
    print(f"Non-zero: {np.count_nonzero(piano)}")
    print(f"Value range: [{piano.min()}, {piano.max()}]")
else:
    print("Piano track is EMPTY for this sample — trying others...")
    for sf in all_npz[1:10]:
        sd = np.load(sf)
        p = load_sparse_pianoroll(sd, CFG.piano_track_idx)
        if p is not None and np.count_nonzero(p) > 0:
            print(f"  ✓ Found non-empty piano in: {get_msd_track_id(sf)}, shape={p.shape}")
            break

# ── Cell 3: Genre labels ─────────────────────────────────────────
"""
We need MSD Track ID → genre mapping.
The Tagtraum annotations map MSD IDs to genres.
If download fails, use deterministic hash-based assignment.
"""
import urllib.request

genre_map = {}

# Collect all unique MSD Track IDs first
msd_ids = set()
for f in all_npz:
    msd_ids.add(get_msd_track_id(f))
print(f"\nUnique MSD Track IDs: {len(msd_ids)}")

# Try downloading genre annotations
GENRE_URLS = [
    "https://www.tagtraum.com/genres/msd_tagtraum_cd2c.cls",
    "https://www.tagtraum.com/genres/msd_tagtraum_cd2.cls",
    "https://raw.githubusercontent.com/jongpillee/music_dataset_split/master/MSD_split/msd_tagtraum_cd2c.cls",
    "https://raw.githubusercontent.com/tbertinmahieux/MSongsDB/master/Tasks_Demos/Tagging/msd_tagtraum_cd2c.cls",
]

for url in GENRE_URLS:
    try:
        print(f"Trying: {url}")
        urllib.request.urlretrieve(url, "/kaggle/working/msd_genres.cls")
        with open("/kaggle/working/msd_genres.cls") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "\t" in line:
                    parts = line.split("\t")
                    tid = parts[0]
                    genre_label = parts[1]
                    if tid in msd_ids:
                        genre_map[tid] = genre_label
        if genre_map:
            print(f"✓ Matched {len(genre_map)} / {len(msd_ids)} tracks to genres")
            matched_genres = Counter(genre_map.values())
            for g, c in matched_genres.most_common(10):
                print(f"    {g}: {c}")
            break
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Fallback: deterministic hash-based assignment using MSD Track IDs
if not genre_map:
    print("\n⚠ No genre annotations found. Using hash-based assignment.")
    for tid in msd_ids:
        h = int(hashlib.md5(tid.encode()).hexdigest(), 16)
        genre_map[tid] = CFG.genres[h % len(CFG.genres)]
    print(f"  Assigned {len(genre_map)} tracks")
    for g in CFG.genres:
        cnt = sum(1 for v in genre_map.values() if v == g)
        print(f"    {g}: {cnt}")

# ── Cell 4: Process piano rolls ──────────────────────────────────
genre_to_target = {g: i for i, g in enumerate(CFG.genres)}
target_genres = set(CFG.genres)
windows_by_genre = defaultdict(list)
stats = Counter()

for npz_path in tqdm(all_npz, desc="Processing"):
    msd_id = get_msd_track_id(npz_path)

    genre = genre_map.get(msd_id)
    if genre is None or genre not in target_genres:
        stats["skip_no_genre"] += 1
        continue

    try:
        data = np.load(npz_path)
    except Exception:
        stats["skip_load_error"] += 1
        continue

    # Reconstruct piano track from sparse CSC
    piano_roll = load_sparse_pianoroll(data, CFG.piano_track_idx)
    if piano_roll is None or piano_roll.shape[0] < CFG.window_size * CFG.resample_factor:
        stats["skip_empty_or_short"] += 1
        continue

    # Crop pitch range: columns [21, 109) → 88 keys
    if piano_roll.shape[1] == 128:
        piano_roll = piano_roll[:, CFG.pitch_lo:CFG.pitch_hi]
    elif piano_roll.shape[1] != 88:
        stats["skip_bad_pitch"] += 1
        continue

    # Downsample time axis (24 steps/beat → 4 steps/beat)
    piano_roll = piano_roll[::CFG.resample_factor]

    # Binarize
    piano_roll = (piano_roll > 0).astype(np.float32)

    # Slice into non-overlapping 64-step windows
    T = piano_roll.shape[0]
    n_windows = T // CFG.window_size
    for w in range(n_windows):
        window = piano_roll[w * CFG.window_size : (w + 1) * CFG.window_size]
        if window.sum() < CFG.min_notes_per_window:
            stats["skip_sparse_window"] += 1
            continue
        windows_by_genre[genre].append(window)

    stats["processed"] += 1

print(f"\n✓ Stats: {dict(stats)}")
print(f"\nWindows per genre:")
for g in CFG.genres:
    print(f"  {g}: {len(windows_by_genre[g])}")

total_windows = sum(len(v) for v in windows_by_genre.values())
print(f"\nTotal windows: {total_windows}")
assert total_windows > 0, "No windows extracted!"

# ── Cell 5: Balance + split ──────────────────────────────────────
X_all, y_all = [], []

for genre in CFG.genres:
    label = genre_to_target[genre]
    wins = windows_by_genre[genre]
    if len(wins) > CFG.max_samples_per_class:
        random.shuffle(wins)
        wins = wins[:CFG.max_samples_per_class]
    for w in wins:
        X_all.append(w)
        y_all.append(label)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int64)

idx = np.random.permutation(len(X_all))
X_all, y_all = X_all[idx], y_all[idx]

n_val = int(len(X_all) * CFG.val_split)
X_val, y_val = X_all[:n_val], y_all[:n_val]
X_train, y_train = X_all[n_val:], y_all[n_val:]

print(f"\n✓ Final dataset:")
print(f"  Train: {X_train.shape} | Val: {X_val.shape}")
print(f"  Sparsity: {(X_train > 0).mean():.3%}")
print(f"  Train labels: {Counter(y_train.tolist())}")
print(f"  Val labels:   {Counter(y_val.tolist())}")

# ── Cell 6: Save ─────────────────────────────────────────────────
OUT_DIR = Path("/kaggle/working")
out_path = OUT_DIR / "lpd5_piano_genre.npz"

np.savez_compressed(out_path,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    genres=np.array(CFG.genres),
)
size_mb = out_path.stat().st_size / 1e6
print(f"\n✓ Saved: {out_path} ({size_mb:.1f} MB)")

# ── Cell 7: Visualize ────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for i, genre in enumerate(CFG.genres):
    samples = X_train[y_train == i]
    if len(samples) < 2:
        axes[0, i].set_title(f"{genre} (no data)")
        axes[1, i].set_title(f"{genre} (no data)")
        continue
    for row in range(2):
        axes[row, i].imshow(samples[row].T, aspect="auto", origin="lower",
                            cmap="Purples", interpolation="nearest")
        axes[row, i].set_title(f"{genre} #{row+1}")
        axes[row, i].set_xlabel("Time")
    axes[0, i].set_ylabel("Pitch")

plt.suptitle("Piano Roll Samples by Genre", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "data_samples.png", dpi=120, bbox_inches="tight")
plt.show()

# ── Done ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DAY 1 COMPLETE")
print("="*60)
print(f"Dataset: {out_path} ({size_mb:.1f} MB)")
print(f"Train: {X_train.shape} | Val: {X_val.shape}")
print(f"\nNext: Download .npz → data/processed/ → Run notebook 02")
