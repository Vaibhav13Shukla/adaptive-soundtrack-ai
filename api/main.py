"""
FastAPI inference backend for Adaptive Soundtrack AI.

Start:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Docker (via docker-compose):
  docker-compose up api
"""
import base64
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from configs.config import DATA, INFER, MODEL, PATHS
from src.ddpm import DDPM
from src.evaluate import evaluate_batch
from src.midi_utils import piano_roll_to_midi_bytes
from src.unet import UNet

logger = logging.getLogger("api")

# ── Global model state ────────────────────────────────────────────
_ddpm:   Optional[DDPM] = None
_device: Optional[str]  = None


def _load_model() -> None:
    global _ddpm, _device
    ckpt_path = PATHS.checkpoints / PATHS.best_ckpt
    if not ckpt_path.exists():
        logger.warning("Checkpoint not found at %s — /generate will return 503.", ckpt_path)
        return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    unet    = UNet(MODEL).to(_device)
    _ddpm   = DDPM(unet, MODEL).to(_device)
    state   = torch.load(str(ckpt_path), map_location=_device, weights_only=True)
    _ddpm.load_state_dict(state["model"])
    _ddpm.eval()
    logger.info("Model loaded on %s  (val_loss=%s)", _device, state.get("val_loss", "—"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Adaptive Soundtrack AI",
    version="0.1.0",
    description="Genre-controlled piano roll generation via Conditional DDPM + CFG + DDIM",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    genre:      str   = Field(..., description=f"One of {DATA.genres}")
    n_samples:  int   = Field(default=1, ge=1, le=4)
    ddim_steps: int   = Field(default=INFER.ddim_steps, ge=10, le=200)
    cfg_scale:  float = Field(default=INFER.cfg_scale,  ge=0.0, le=10.0)


class SampleResult(BaseModel):
    midi_b64: str             # base64-encoded MIDI bytes
    roll:     List[List[int]] # (64, 88) binary piano roll as nested list
    stats:    Dict[str, float]


class GenerateResponse(BaseModel):
    genre:   str
    samples: List[SampleResult]


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _ddpm is not None, "device": _device}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _ddpm is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Checkpoint missing.")
    if req.genre not in DATA.genres:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown genre '{req.genre}'. Valid genres: {DATA.genres}",
        )

    genre_idx = DATA.genres.index(req.genre)
    g         = torch.full((req.n_samples,), genre_idx, device=_device).long()
    shape     = (req.n_samples, 1, DATA.window_size, 88)

    with torch.no_grad():
        x = _ddpm.ddim_sample(
            shape, g,
            ddim_steps=req.ddim_steps,
            cfg_scale=req.cfg_scale,
            eta=0.0,
        )

    rolls   = (x.squeeze(1).cpu().numpy() > 0.0).astype(np.float32)  # (B, 64, 88)
    metrics = evaluate_batch(rolls)

    samples = []
    for roll in rolls:
        midi_bytes = piano_roll_to_midi_bytes(roll, pitch_lo=DATA.pitch_lo)
        samples.append(SampleResult(
            midi_b64=base64.b64encode(midi_bytes).decode(),
            roll=roll.astype(int).tolist(),
            stats=metrics,
        ))

    return GenerateResponse(genre=req.genre, samples=samples)
