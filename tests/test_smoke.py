"""
Smoke tests — verify all modules import and core logic runs correctly
without requiring a trained checkpoint.
"""
import numpy as np
import pytest
import torch

from configs.config import DATA, MODEL, PATHS, TRAIN
from src.dataset import PianoRollDataset
from src.ddpm import DDPM
from src.evaluate import (
    evaluate_batch,
    note_density,
    pitch_entropy,
    scale_consistency,
)
from src.midi_utils import piano_roll_to_image, piano_roll_to_midi_bytes
from src.unet import UNet


# ── Config ────────────────────────────────────────────────────────

def test_config_defaults():
    assert len(DATA.genres) == 4
    assert MODEL.T == 1000
    assert MODEL.beta_schedule == "cosine"
    assert TRAIN.batch_size == 64


def test_paths_repo_root_exists():
    """PathConfig.repo_root must resolve to an actual directory, never CWD blindly."""
    assert PATHS.repo_root.is_dir(), f"repo_root {PATHS.repo_root} is not a directory"


# ── Model ─────────────────────────────────────────────────────────

def test_unet_forward():
    model = UNet(MODEL)
    model.eval()
    B = 2
    x = torch.randn(B, 1, 64, 88)
    t = torch.randint(0, MODEL.T, (B,))
    g = torch.randint(0, MODEL.n_genres, (B,))
    with torch.no_grad():
        out = model(x, t, g)
    assert out.shape == (B, 1, 64, 88), f"Bad output shape: {out.shape}"


def test_unet_unconditional():
    """Model must forward without a genre label (g=None)."""
    model = UNet(MODEL)
    model.eval()
    B = 2
    x = torch.randn(B, 1, 64, 88)
    t = torch.randint(0, MODEL.T, (B,))
    with torch.no_grad():
        out = model(x, t, g=None)
    assert out.shape == (B, 1, 64, 88)


def test_ddpm_loss_finite():
    model = UNet(MODEL)
    ddpm  = DDPM(model)
    B     = 2
    x0    = torch.randn(B, 1, 64, 88)
    g     = torch.randint(0, MODEL.n_genres, (B,))
    loss  = ddpm.p_losses(x0, g)
    assert loss.item() > 0
    assert torch.isfinite(loss), "Training loss must be finite"


def test_ddpm_cosine_schedule_monotone():
    """Cosine alpha_bars must decrease monotonically from ~1 to ~0."""
    model = UNet(MODEL)
    ddpm  = DDPM(model)
    ab    = ddpm.alpha_bars.cpu().numpy()
    assert (np.diff(ab) < 0).all(), "alpha_bars must be strictly decreasing"


def test_ddpm_ddim_no_nan():
    """DDIM sample (eta=0) must produce finite outputs."""
    model = UNet(MODEL)
    ddpm  = DDPM(model)
    ddpm.eval()
    g = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        x = ddpm.ddim_sample((1, 1, 64, 88), g, ddim_steps=5, eta=0.0)
    assert torch.isfinite(x).all(), "DDIM output contains NaN/Inf"


def test_ddpm_ddim_eta_no_nan():
    """DDIM sample with eta>0 must not produce NaN (regression for sigma² clamp)."""
    model = UNet(MODEL)
    ddpm  = DDPM(model)
    ddpm.eval()
    g = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        x = ddpm.ddim_sample((1, 1, 64, 88), g, ddim_steps=5, eta=1.0)
    assert torch.isfinite(x).all(), "DDIM output with eta=1.0 contains NaN/Inf"


# ── Augmentation ──────────────────────────────────────────────────

def test_augmentation_no_wrap():
    """Shifted regions must contain -1 (silence), not values from the opposite boundary."""
    X = np.zeros((4, 64, 88), dtype=np.float32)
    X[:, 0, 40] = 1.0    # single note at t=0, pitch=40
    y = np.zeros(4, dtype=np.int64)
    ds = PianoRollDataset(X, y, augment=True)

    torch.manual_seed(123)
    for _ in range(40):
        roll, _ = ds[0]   # (1, 64, 88)
        assert torch.isfinite(roll).all(), "Augmented roll contains NaN/Inf"
        assert roll.min() >= -1.0 - 1e-5, "Augmented roll underflows -1"
        assert roll.max() <=  1.0 + 1e-5, "Augmented roll overflows +1"


def test_augmentation_output_shape():
    X = np.zeros((4, 64, 88), dtype=np.float32)
    y = np.zeros(4, dtype=np.int64)
    ds = PianoRollDataset(X, y, augment=True)
    roll, label = ds[0]
    assert roll.shape == (1, 64, 88)
    assert label.shape == ()


# ── Metrics ───────────────────────────────────────────────────────

def test_evaluate_batch_keys():
    rolls = np.random.randint(0, 2, (4, 64, 88)).astype(np.float32)
    metrics = evaluate_batch(rolls)
    assert set(metrics.keys()) == {
        "pitch_entropy", "note_density", "scale_consistency", "empty_ratio"
    }


def test_scale_consistency_pentatonic():
    """A pure pentatonic pattern must score ≥ 0.9 with the extended scale check."""
    roll = np.zeros((64, 88), dtype=np.float32)
    for pc in [0, 2, 4, 7, 9]:          # C-major pentatonic pitch classes
        roll[:, pc + 36] = 1.0
    score = scale_consistency(roll)
    assert score >= 0.9, f"Pentatonic roll should score ≥ 0.9, got {score:.3f}"


def test_scale_consistency_minor():
    """A natural-minor pattern must also score well."""
    roll = np.zeros((64, 88), dtype=np.float32)
    for pc in [0, 2, 3, 5, 7, 8, 10]:   # A natural minor pitch classes
        roll[:, pc + 36] = 1.0
    score = scale_consistency(roll)
    assert score >= 0.9, f"Minor roll should score ≥ 0.9, got {score:.3f}"


def test_pitch_entropy_silent():
    assert pitch_entropy(np.zeros((64, 88), dtype=np.float32)) == 0.0


def test_note_density_half():
    roll = np.zeros((64, 88), dtype=np.float32)
    roll[::2, 40] = 1.0   # every other step active
    assert abs(note_density(roll) - 0.5) < 0.01


# ── MIDI utils ────────────────────────────────────────────────────

def test_midi_bytes_nonempty():
    roll = np.random.randint(0, 2, (64, 88)).astype(np.float32)
    data = piano_roll_to_midi_bytes(roll)
    assert isinstance(data, bytes) and len(data) > 0


def test_midi_bytes_silent_roll():
    """Silent roll → valid (near-empty) MIDI file."""
    data = piano_roll_to_midi_bytes(np.zeros((64, 88), dtype=np.float32))
    assert isinstance(data, bytes) and len(data) > 0


def test_piano_roll_to_image_writeable():
    """Returned image array must be writeable (not a negative-stride view)."""
    roll = np.random.randint(0, 2, (64, 88)).astype(np.float32)
    img  = piano_roll_to_image(roll)
    assert img.flags["WRITEABLE"], "piano_roll_to_image must return a writeable array"
    assert img.shape == (88, 64), f"Expected (88, 64), got {img.shape}"
