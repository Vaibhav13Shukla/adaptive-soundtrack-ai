# Adaptive Soundtrack AI

**Conditional Diffusion → Genre-Controlled MIDI**

Generate editable piano rolls conditioned on musical genre using a class-conditional DDPM. Trained on the Lakh Pianoroll Dataset. Outputs are MIDI files playable in any DAW.


---

## Demo

```bash
streamlit run app.py
```

Pick a genre → generate → download MIDI.

---

## What & Why

Music generation tools (Suno, Riffusion, Stable Audio) output raw audio waveforms — opaque, expensive, not editable. We generate **symbolic music** (MIDI) using diffusion. Output is a piano roll you can drop into FL Studio, Ableton, or Unity.

Not competing with Suno. Different layer of the stack: editable structure, not frozen audio.

---

## Course Concepts Applied

| Concept | Where in code |
|---|---|
| Manifold hypothesis | `src/ddpm.py` (forward diffusion comments) |
| Forward diffusion q(x_t \| x_{t-1}) | `src/ddpm.py:q_sample` |
| Reverse diffusion / denoising | `src/ddpm.py:p_losses` |
| U-Net noise prediction | `src/unet.py` |
| Sinusoidal time embedding | `src/unet.py:SinusoidalPosEmb` |
| Class-conditional generation | `src/unet.py:TimeGenreEmbedding` |
| Classifier-free guidance | `src/ddpm.py:p_losses` (dropout), `ddim_sample` (interpolation) |
| Score function ∇_x log q(x_t) | `src/ddpm.py` (referenced in DDIM sampler) |
| DDIM fast sampling | `src/ddpm.py:ddim_sample` |

---

## Quickstart

```bash
git clone https://github.com/YOUR_USER/adaptive-soundtrack-ai
cd adaptive-soundtrack-ai
pip install -r requirements.txt

# Download trained checkpoint from Kaggle (manual)
# Place in: checkpoints/checkpoint_best.pt

# Run demo
streamlit run app.py
```

---

## Dataset

**Lakh Pianoroll Dataset** (LPD-5-cleansed)
- 21,425 five-track piano rolls
- 4/4 time signature only
- Genre labels via Million Song Dataset matching
- We use the Piano track; binarize, slice into 64-step windows

Available on Kaggle: `cloudoak/lpd-5-cleansed`

---

## Project Structure

```
configs/config.py          ← All hyperparameters
src/                       ← Library code
  dataset.py               ← PyTorch Dataset + DataLoaders
  unet.py                  ← Conditional U-Net
  ddpm.py                  ← Diffusion engine + DDIM sampling
  trainer.py               ← Training loop + checkpointing
  midi_utils.py            ← Piano roll ↔ MIDI conversion
  evaluate.py              ← Quantitative metrics
notebooks/                 ← One per day, run on Kaggle in order
train.py                   ← Single command training
infer.py                   ← Generate samples
app.py                     ← Streamlit demo
```

---

## Results

(Updated after Day 4)

| Metric | Unconditional | Conditional (CFG=3.0) |
|---|---|---|
| Val Loss | TBD | TBD |
| Pitch Entropy | TBD | TBD |
| Scale Consistency | TBD | TBD |
| Genre Classifier Acc | — | TBD |

---

## Acknowledgements

- Lakh Pianoroll Dataset: Dong et al., AAAI 2018
- DDPM: Ho et al., NeurIPS 2020
- Classifier-Free Guidance: Ho & Salimans, 2022
- DDIM: Song et al., ICLR 2021
