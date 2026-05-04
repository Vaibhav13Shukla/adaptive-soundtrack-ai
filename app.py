"""
Streamlit demo — Adaptive Soundtrack AI
Run: streamlit run app.py
"""
import time
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from configs.config import MODEL, INFER, DATA, PATHS
from src.unet       import UNet
from src.ddpm       import DDPM
from src.midi_utils import piano_roll_to_midi_bytes

# ── Page ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Soundtrack AI",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0e0e0e; }
  .block-container { padding-top: 2rem; }
  .metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
  }
  .genre-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px;
  }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    ckpt_path = PATHS.checkpoints / PATHS.best_ckpt
    if not ckpt_path.exists():
        return None, None, "not_found"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        unet  = UNet(MODEL).to(device)
        ddpm  = DDPM(unet, MODEL).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        ddpm.load_state_dict(state["model"])
        ddpm.eval()
        val_loss = state.get("val_loss", "—")
        return ddpm, device, val_loss
    except Exception as e:
        return None, None, str(e)

with st.spinner("Loading model..."):
    ddpm, device, val_loss = load_model()

# ── Header ────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("## 🎹 Adaptive Soundtrack AI")
    st.caption("Conditional DDPM · Genre-controlled piano roll generation · Generative AI Course Project")
with col_status:
    if ddpm is not None:
        st.success("Model loaded")
        st.caption(f"Val loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"Val loss: {val_loss}")
    else:
        st.error("No checkpoint found")
        st.caption("Download from Kaggle → checkpoints/checkpoint_best.pt")
        st.stop()

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Generate")

    genre = st.selectbox(
        "Genre",
        DATA.genres,
        help="The genre label the model conditions on"
    )

    cfg_scale = st.slider(
        "CFG guidance scale",
        min_value=0.0, max_value=7.0,
        value=3.0, step=0.5,
        help="0 = unconditional random. Higher = stronger genre adherence."
    )

    n_samples = st.radio("Samples", [1, 2, 4], index=1, horizontal=True)

    ddim_steps = st.select_slider(
        "DDIM steps",
        options=[10, 20, 50, 100],
        value=50,
        help="More steps = higher quality, slower generation"
    )

    st.divider()
    st.caption("**How it works**")
    st.caption(
        "Pure Gaussian noise is denoised "
        "step by step using a U-Net guided "
        "by a genre embedding (classifier-free "
        "guidance). The output is a binary "
        "88×64 piano roll converted to MIDI."
    )

    generate_btn = st.button(
        "🎲 Generate",
        type="primary",
        use_container_width=True
    )

# ── Main area ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Generate", "How it works", "Metrics"])

with tab1:
    if generate_btn:
        progress = st.progress(0, text="Starting DDIM sampling...")
        t0 = time.time()

        genre_idx = DATA.genres.index(genre)
        g = torch.full((n_samples,), genre_idx, device=device).long()
        shape = (n_samples, 1, DATA.window_size, 88)

        with torch.no_grad():
            progress.progress(10, text=f"Running DDIM ({ddim_steps} steps)...")
            x = ddpm.ddim_sample(
                shape, g,
                ddim_steps=ddim_steps,
                cfg_scale=cfg_scale,
                eta=0.0,
            )

        # Decode: [-1,1] → binary
        rolls = (x.squeeze(1).cpu().numpy() > 0.0).astype(np.float32)
        elapsed = time.time() - t0

        progress.progress(80, text="Rendering piano rolls...")

        # Stats per roll
        def roll_stats(roll):
            notes = roll.sum()
            if notes == 0:
                return {"notes": 0, "density": 0.0,
                        "pitch_range": "—", "entropy": 0.0}
            pitch_hist = roll.sum(axis=0)  # (88,)
            active_pitches = np.where(pitch_hist > 0)[0]
            lo, hi = active_pitches.min(), active_pitches.max()
            prob = pitch_hist / (pitch_hist.sum() + 1e-8)
            entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0] + 1e-8))
            return {
                "notes": int(notes),
                "density": float(notes / roll.size),
                "pitch_range": f"MIDI {lo + DATA.pitch_lo}–{hi + DATA.pitch_lo}",
                "entropy": float(entropy),
            }

        # Layout
        cols = st.columns(n_samples)
        for i, (col, roll) in enumerate(zip(cols, rolls)):
            stats = roll_stats(roll)
            with col:
                # Piano roll plot
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor("#1a1a1a")
                ax.set_facecolor("#0e0e0e")
                ax.imshow(
                    roll.T,
                    aspect="auto",
                    origin="lower",
                    cmap="Purples",
                    interpolation="nearest",
                    vmin=0, vmax=1,
                )
                ax.set_title(
                    f"{genre} #{i+1}",
                    color="white", fontsize=12, pad=8
                )
                ax.set_xlabel("Time (steps)", color="#888")
                ax.set_ylabel("Pitch index", color="#888")
                ax.tick_params(colors="#888")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # Quick stats
                s1, s2 = st.columns(2)
                s1.metric("Notes", stats["notes"])
                s2.metric("Density", f"{stats['density']:.2%}")
                s1.metric("Pitch range", stats["pitch_range"])
                s2.metric("Entropy", f"{stats['entropy']:.2f}")

                # MIDI export — generate in memory (no shared filesystem state)
                midi_bytes = piano_roll_to_midi_bytes(roll, pitch_lo=DATA.pitch_lo)
                st.download_button(
                    label=f"⬇ Download MIDI {i+1}",
                    data=midi_bytes,
                    file_name=f"{genre}_{i+1}.mid",
                    mime="audio/midi",
                    use_container_width=True,
                    key=f"dl_{i}",
                )

        progress.progress(100, text=f"Done in {elapsed:.1f}s")
        st.success(
            f"Generated {n_samples} {genre} sample(s) "
            f"in {elapsed:.1f}s using {ddim_steps} DDIM steps."
        )
    else:
        # Placeholder
        st.markdown("""
        <div style='text-align:center; padding: 4rem 0; color: #666;'>
            <div style='font-size: 48px; margin-bottom: 1rem;'>🎹</div>
            <div style='font-size: 18px; margin-bottom: 0.5rem;'>Pick a genre and hit Generate</div>
            <div style='font-size: 14px;'>Each sample is a 4-bar piano roll generated from pure noise</div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Architecture")
    st.markdown("""
**Input:** Pure Gaussian noise `(B, 1, 64, 88)` — identical to how the training forward diffusion corrupts real piano rolls to noise at `T=1000`.

**Conditioning:** Genre label → learned embedding → added to sinusoidal timestep embedding → injected via FiLM (scale + shift) into every U-Net ResBlock.

**Classifier-Free Guidance:** During training, genre is randomly nulled out (15% of batches). At inference, the model runs twice per step — once conditional, once unconditional — and the outputs are interpolated:

```
ε_guided = ε_uncond + cfg_scale × (ε_cond - ε_uncond)
```

**DDIM Sampling:** Instead of 1000 denoising steps, DDIM deterministically skips to 50 steps with identical output quality. This is what makes real-time generation possible.

**Output:** Piano roll binarized at threshold 0 → converted to MIDI via `pretty_midi`.
    """)
    st.markdown("### Course concepts applied")
    st.markdown("""
| Concept | Where |
|---|---|
| Manifold hypothesis | Piano rolls live on a thin manifold in 88×64 space. Most random binary matrices are not music. |
| Forward diffusion q(x_t \\| x_0) | `src/ddpm.py:q_sample` — reparameterization trick |
| Reverse diffusion | `src/ddpm.py:p_losses` — U-Net predicts ε |
| U-Net + skip connections | `src/unet.py` |
| Sinusoidal timestep embedding | `src/unet.py:SinusoidalPosEmb` |
| Class conditioning | `src/unet.py:TimeGenreEmbedding` |
| Classifier-free guidance | `src/ddpm.py` — training dropout + inference interpolation |
| DDIM fast sampling | `src/ddpm.py:ddim_sample` |
    """)

with tab3:
    # Load training log if it exists
    import json
    log_path = PATHS.logs / "train_log.json"
    if log_path.exists():
        with open(log_path) as f:
            records = json.load(f)

        epochs      = [r["epoch"] for r in records]
        train_losses = [r["train_loss"] for r in records]
        val_losses   = [r["val_loss"] for r in records]

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total epochs", len(records))
        m2.metric("Best val loss", f"{min(val_losses):.4f}")
        m3.metric("Final train loss", f"{train_losses[-1]:.4f}")
        m4.metric("Final val loss",   f"{val_losses[-1]:.4f}")

        # Loss curve
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#0e0e0e")
        ax.plot(epochs, train_losses, color="#7C3AED",
                linewidth=2, label="Train loss")
        ax.plot(epochs, val_losses, color="#A78BFA",
                linewidth=2, linestyle="--", label="Val loss")
        ax.set_xlabel("Epoch", color="#888")
        ax.set_ylabel("MSE Loss", color="#888")
        ax.set_title("Training curve", color="white")
        ax.tick_params(colors="#888")
        ax.legend(facecolor="#1a1a1a", labelcolor="white",
                  framealpha=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info(
            "No training log found at `logs/train_log.json`.\n\n"
            "Run training on Kaggle and push logs/ to GitHub, "
            "then pull locally to see metrics here."
        )
