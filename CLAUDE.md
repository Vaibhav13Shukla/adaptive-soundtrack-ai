# CLAUDE.md — Project Context

## What this is
Conditional DDPM for genre-controlled piano roll generation.
4-day sprint. Kaggle P100 for training. Streamlit for demo.

## Architecture
- Input: (B, 1, 64, 88) piano roll (1 channel, 64 timesteps, 88 piano keys)
- Model: 2D U-Net + sinusoidal time embedding + genre embedding (CFG)
- Output: predicted noise ε
- Inference: DDIM 50 steps

## Constraints
- Kaggle P100 free tier: 16GB GPU, 9hr session, 12GB RAM
- batch_size <= 64
- Checkpoint every 5 epochs to GitHub
- No fancy libraries — PyTorch + NumPy only

## File map
configs/config.py  → ALL hyperparameters
src/dataset.py     → Piano roll loading + augmentation
src/unet.py        → Conditional U-Net architecture
src/ddpm.py        → Forward/reverse diffusion + CFG
src/trainer.py     → Training loop + checkpointing
src/midi_utils.py  → Piano roll ↔ MIDI conversion
src/evaluate.py    → Metrics (entropy, scale consistency, classifier acc)
src/logger.py      → JSON experiment logging
train.py           → Single entry point — runs full training
infer.py           → Generate MIDI from trained model
app.py             → Streamlit demo

## Critical rules
- Never commit data files (*.npz, *.tar.gz)
- Never commit large checkpoints (>50MB → use Git LFS)
- Always git pull before Kaggle session
- Always git push at end of Kaggle session
- Print tensor shapes everywhere — Karpathy rule
