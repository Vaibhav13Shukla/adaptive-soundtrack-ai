"""
Streamlit demo. Run locally: streamlit run app.py
"""
import sys, io, torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(".")

from configs.config import MODEL, INFER, DATA, PATHS
from src.unet       import UNet
from src.ddpm       import DDPM
from src.midi_utils import piano_roll_to_midi


# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Soundtrack AI",
    page_icon="🎹",
    layout="wide",
)

st.title("🎹 Adaptive Soundtrack AI")
st.caption("Conditional Diffusion → Genre-Controlled MIDI")


# ── Load model (cached) ──────────────────────────────────────────
@st.cache_resource
def load_model():
    ckpt_path = PATHS.checkpoints / PATHS.best_ckpt
    if not ckpt_path.exists():
        st.error(f"Checkpoint not found: {ckpt_path}\n"
                 "Download from Kaggle and place in checkpoints/")
        st.stop()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = UNet(MODEL).to(device)
    ddpm = DDPM(unet, MODEL).to(device)
    state = torch.load(ckpt_path, map_location=device)
    ddpm.load_state_dict(state["model"])
    ddpm.eval()
    return ddpm, device


with st.spinner("Loading model..."):
    ddpm, device = load_model()


# ── Sidebar controls ──────────────────────────────────────────────
st.sidebar.header("Generate")
genre = st.sidebar.selectbox("Genre", DATA.genres)
cfg_scale = st.sidebar.slider("Guidance scale (CFG)", 0.0, 7.0, 3.0, 0.5,
                               help="Higher = stronger genre adherence")
n_samples = st.sidebar.slider("Number of samples", 1, 4, 2)
ddim_steps = st.sidebar.slider("DDIM steps", 10, 100, 50, 10,
                                help="More steps = better quality, slower")

go = st.sidebar.button("🎲 Generate", type="primary", use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────
st.markdown(f"""
**How it works:** A conditional U-Net learns to denoise pure Gaussian noise into
musically coherent piano rolls. Guided by the genre embedding via classifier-free
guidance with scale = `{cfg_scale}`.

Pick a genre on the left, hit generate, download MIDI.
""")

if go:
    with st.spinner(f"Generating {n_samples} {genre} samples..."):
        genre_idx = DATA.genres.index(genre)
        g = torch.full((n_samples,), genre_idx, device=device).long()
        shape = (n_samples, 1, DATA.window_size, 88)
        x = ddpm.ddim_sample(shape, g, ddim_steps=ddim_steps,
                             cfg_scale=cfg_scale, eta=0.0)
        rolls = (x.squeeze(1).cpu().numpy() > 0.0).astype(np.float32)

    cols = st.columns(n_samples)
    for i, (col, roll) in enumerate(zip(cols, rolls)):
        with col:
            # Plot
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(roll.T, aspect="auto", origin="lower",
                      cmap="Purples", interpolation="nearest")
            ax.set_title(f"{genre} #{i+1}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Pitch")
            st.pyplot(fig)
            plt.close(fig)

            # MIDI download
            tmp = PATHS.midi_examples / f"streamlit_{genre}_{i}.mid"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            piano_roll_to_midi(roll, str(tmp), pitch_lo=DATA.pitch_lo)
            with open(tmp, "rb") as f:
                st.download_button(
                    f"⬇️ Download MIDI {i+1}",
                    f.read(),
                    file_name=f"{genre}_{i+1}.mid",
                    mime="audio/midi",
                    key=f"dl_{i}",
                )

    st.success(f"Generated {n_samples} samples. Open the MIDI in any DAW.")
else:
    st.info("👈 Pick a genre and hit Generate.")
