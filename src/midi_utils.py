"""
Piano roll ↔ MIDI conversion.
"""
import io
import numpy as np
import pretty_midi


def _build_pretty_midi(
    roll: np.ndarray,
    pitch_lo: int = 21,
    fs: int = 8,
    velocity: int = 80,
) -> pretty_midi.PrettyMIDI:
    """Internal helper: build a PrettyMIDI object from a binary piano roll.

    Args:
        roll: (T, 88) array with values in {0, 1}.  Values > 0 are treated as
              active.  Do NOT pass the raw model output in [-1, 1]; binarize
              first with ``(raw > 0).astype(float)``.
        pitch_lo: MIDI pitch number for column 0 (default 21 = A0).
        fs: frames per second (default 8 = 4 steps/beat × 120 bpm).
        velocity: MIDI velocity for every note (0–127).
    """
    T, P = roll.shape
    assert P == 88, f"Expected 88 pitches, got {P}"

    pm   = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)   # 0 = Acoustic Grand Piano
    step = 1.0 / fs

    for pitch_idx in range(P):
        actual_pitch = pitch_idx + pitch_lo
        col          = roll[:, pitch_idx]

        # Threshold at 0 — works correctly for binary {0, 1} float input
        i = 0
        while i < T:
            if col[i] > 0.0:
                start = i
                while i < T and col[i] > 0.0:
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
    return pm


def piano_roll_to_midi(
    roll: np.ndarray,           # (T, 88) binary float32 — values 0.0 or 1.0
    out_path: str,
    pitch_lo: int = 21,
    fs: int = 8,                # frames per second (4 steps/beat × 120bpm = 8fps)
    velocity: int = 80,
) -> str:
    """Convert a binary piano roll → MIDI file written to *out_path*.

    ``roll`` must be a (T, 88) array with values in {0, 1}.  Binarize raw
    model output before calling: ``(raw > 0).astype(float)``.
    """
    pm = _build_pretty_midi(roll, pitch_lo=pitch_lo, fs=fs, velocity=velocity)
    pm.write(str(out_path))
    return str(out_path)


def piano_roll_to_midi_bytes(
    roll: np.ndarray,           # (T, 88) binary float32 — values 0.0 or 1.0
    pitch_lo: int = 21,
    fs: int = 8,
    velocity: int = 80,
) -> bytes:
    """Like *piano_roll_to_midi* but returns raw MIDI bytes instead of writing a file.

    Safe for concurrent use — no shared filesystem state.
    """
    pm  = _build_pretty_midi(roll, pitch_lo=pitch_lo, fs=fs, velocity=velocity)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def piano_roll_to_image(roll: np.ndarray) -> np.ndarray:
    """Convert (T, 88) piano roll → image-ready (88, T) uint8 for display.

    Origin is lower so low pitches appear at the bottom.
    Returns a writeable C-contiguous array (safe for downstream modification).
    """
    return np.ascontiguousarray((roll.T * 255).astype(np.uint8)[::-1])
