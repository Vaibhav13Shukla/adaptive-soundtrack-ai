"""
Piano roll ↔ MIDI conversion.
"""
import numpy as np
import pretty_midi
from pathlib import Path


def piano_roll_to_midi(
    roll: np.ndarray,           # (T, 88) binary
    out_path: str,
    pitch_lo: int = 21,
    fs: int = 8,                # frames per second (4 steps/beat × 120bpm = 8fps)
    velocity: int = 80,
):
    """
    Convert a binary piano roll → MIDI file.
    Each timestep that has a '1' for a pitch becomes a note-on.
    Consecutive 1s are merged into single notes.
    """
    T, P = roll.shape
    assert P == 88, f"Expected 88 pitches, got {P}"

    pm    = pretty_midi.PrettyMIDI()
    inst  = pretty_midi.Instrument(program=0)   # 0 = Acoustic Grand Piano
    step  = 1.0 / fs

    for pitch_idx in range(P):
        actual_pitch = pitch_idx + pitch_lo
        col          = roll[:, pitch_idx]

        # Find note runs: consecutive 1s
        i = 0
        while i < T:
            if col[i] > 0.5:
                start = i
                while i < T and col[i] > 0.5:
                    i += 1
                end = i
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=actual_pitch,
                    start=start * step,
                    end=end * step,
                )
                inst.notes.append(note)
            else:
                i += 1

    pm.instruments.append(inst)
    pm.write(str(out_path))
    return out_path


def piano_roll_to_image(roll: np.ndarray) -> np.ndarray:
    """
    Convert (T, 88) piano roll → image-ready (88, T) uint8 for display.
    Origin lower so low pitches are at the bottom.
    """
    return (roll.T * 255).astype(np.uint8)[::-1]
