"""
Evaluation metrics for generated piano rolls.
"""
import numpy as np
from typing import Dict, List
from collections import Counter


def pitch_entropy(roll: np.ndarray) -> float:
    """
    Shannon entropy of pitch distribution.
    Higher = more diverse pitch usage.
    roll: (T, 88) binary
    """
    pitch_counts = roll.sum(axis=0)  # (88,)
    total = pitch_counts.sum()
    if total == 0:
        return 0.0
    probs = pitch_counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def note_density(roll: np.ndarray) -> float:
    """Average number of active notes per timestep."""
    return roll.sum(axis=1).mean()


def scale_consistency(roll: np.ndarray) -> float:
    """
    Fraction of notes that belong to the most common scale.
    Uses pitch classes (mod 12). Higher = more tonal.
    """
    active_pitches = np.where(roll.sum(axis=0) > 0)[0]
    if len(active_pitches) == 0:
        return 0.0

    pitch_classes = active_pitches % 12

    # Major scale patterns (all 12 rotations)
    major = {0, 2, 4, 5, 7, 9, 11}
    best_match = 0.0
    for root in range(12):
        scale = {(p + root) % 12 for p in major}
        match = sum(1 for pc in pitch_classes if pc in scale) / len(pitch_classes)
        best_match = max(best_match, match)

    return best_match


def empty_ratio(roll: np.ndarray) -> float:
    """Fraction of timesteps with zero notes."""
    return (roll.sum(axis=1) == 0).mean()


def evaluate_batch(rolls: np.ndarray) -> Dict[str, float]:
    """
    Compute all metrics over a batch of piano rolls.
    rolls: (B, T, 88) binary
    Returns dict of mean metrics.
    """
    metrics = {
        "pitch_entropy": [],
        "note_density": [],
        "scale_consistency": [],
        "empty_ratio": [],
    }
    for roll in rolls:
        metrics["pitch_entropy"].append(pitch_entropy(roll))
        metrics["note_density"].append(note_density(roll))
        metrics["scale_consistency"].append(scale_consistency(roll))
        metrics["empty_ratio"].append(empty_ratio(roll))

    return {k: float(np.mean(v)) for k, v in metrics.items()}
