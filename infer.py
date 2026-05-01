"""
Generate piano rolls + MIDI from trained checkpoint.

Usage:
  python infer.py --genre Jazz --n_samples 4
"""
import argparse, sys, torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(".")

from configs.config import MODEL, INFER, DATA, PATHS
from src.unet       import UNet
from src.ddpm       import DDPM
from src.midi_utils import piano_roll_to_midi


def load_model(ckpt_path: str, device: str):
    unet = UNet(MODEL).to(device)
    ddpm = DDPM(unet, MODEL).to(device)
    state = torch.load(ckpt_path, map_location=device)
    ddpm.load_state_dict(state["model"])
    ddpm.eval()
    return ddpm


def generate(ddpm: DDPM, genre_idx: int, n_samples: int, device: str):
    g = torch.full((n_samples,), genre_idx, device=device).long()
    shape = (n_samples, 1, DATA.window_size, 88)
    x = ddpm.ddim_sample(shape, g,
                         ddim_steps=INFER.ddim_steps,
                         cfg_scale=INFER.cfg_scale,
                         eta=INFER.ddim_eta)
    # (B, 1, 64, 88) -> (B, 64, 88) binary
    rolls = (x.squeeze(1).cpu().numpy() > 0.0).astype(np.float32)
    return rolls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--genre", required=True, choices=DATA.genres)
    p.add_argument("--n_samples", type=int, default=INFER.n_samples)
    p.add_argument("--ckpt", default=str(PATHS.checkpoints / PATHS.best_ckpt))
    args = p.parse_args()

    PATHS.make_dirs()
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    genre_idx = DATA.genres.index(args.genre)

    print(f"Loading model from {args.ckpt}")
    ddpm = load_model(args.ckpt, device)

    print(f"Generating {args.n_samples} {args.genre} samples...")
    rolls = generate(ddpm, genre_idx, args.n_samples, device)

    # Save MIDI + visual
    fig, axes = plt.subplots(1, args.n_samples, figsize=(4*args.n_samples, 4))
    if args.n_samples == 1:
        axes = [axes]
    for i, roll in enumerate(rolls):
        midi_path = PATHS.midi_examples / f"generated_{args.genre}_{i}.mid"
        piano_roll_to_midi(roll, str(midi_path),
                           pitch_lo=DATA.pitch_lo)
        axes[i].imshow(roll.T, aspect="auto", origin="lower",
                       cmap="Purples", interpolation="nearest")
        axes[i].set_title(f"{args.genre} {i+1}")
        axes[i].set_xlabel("Time"); axes[i].set_ylabel("Pitch")

    plot_path = PATHS.plots / f"generated_{args.genre}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n✓ Saved {args.n_samples} MIDI files → {PATHS.midi_examples}")
    print(f"✓ Plot → {plot_path}")


if __name__ == "__main__":
    main()
