"""
Central configuration. All hyperparameters live here.
Change values here only. Never hardcode in notebooks/scripts.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _default_repo_root() -> Path:
    """Resolve repo root from this file's location, independent of CWD."""
    return Path(__file__).parent.parent.resolve()


@dataclass
class DataConfig:
    # Genre selection — top 4 with most samples in LPD
    genres: List[str] = field(
        default_factory=lambda: ["Pop_Rock", "Electronic", "Rap", "Jazz"]
    )

    # Piano roll preprocessing
    pitch_lo: int = 21          # MIDI 21 = A0 (lowest piano key)
    pitch_hi: int = 109         # exclusive upper bound: range(21, 109) gives 88 MIDI pitches (21–108)
    resample_factor: int = 6    # 24 steps/beat → 4 steps/beat (16th notes)
    window_size: int = 64       # 64 timesteps = 4 bars at 4/4
    piano_track_idx: int = 1    # LPD-5: 0=Drums, 1=Piano, 2=Guitar, 3=Bass, 4=Strings
    min_notes_per_window: int = 4

    # Balancing + split
    max_samples_per_class: int = 6000
    val_split: float = 0.1
    seed: int = 42


@dataclass
class ModelConfig:
    # U-Net
    in_channels: int = 1
    base_channels: int = 64
    channel_mults: List[int] = field(default_factory=lambda: [1, 2, 4])
    n_res_blocks: int = 2
    dropout: float = 0.1
    n_genres: int = 4           # null token added internally for CFG
    attn_heads: int = 4         # self-attention heads at U-Net bottleneck

    # Diffusion schedule
    T: int = 1000
    beta_schedule: str = "cosine"   # "cosine" (recommended) or "linear"
    beta_start: float = 1e-4        # used only when beta_schedule="linear"
    beta_end: float = 0.02          # used only when beta_schedule="linear"

    # CFG
    cfg_dropout: float = 0.15   # prob of nulling genre during training
    cfg_scale: float = 3.0      # guidance strength at inference


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 2e-4
    lr_min: float = 1e-6        # cosine-annealing floor
    n_epochs: int = 60          # 60 epochs reasonable for 4-day sprint
    save_every: int = 5         # checkpoint every 5 epochs
    sample_every: int = 10      # generate samples every 10 epochs
    grad_clip: float = 1.0
    num_workers: int = 2
    resume: bool = True
    use_amp: bool = True        # mixed-precision training (CUDA only)


@dataclass
class InferConfig:
    ddim_steps: int = 50
    ddim_eta: float = 0.0       # 0 = deterministic
    cfg_scale: float = 3.0
    n_samples: int = 4


@dataclass
class PathConfig:
    """All paths used across the project."""
    repo_root: Path = field(default_factory=_default_repo_root)

    @property
    def data_processed(self) -> Path:
        return self.repo_root / "data" / "processed"

    @property
    def checkpoints(self) -> Path:
        return self.repo_root / "checkpoints"

    @property
    def plots(self) -> Path:
        return self.repo_root / "plots"

    @property
    def outputs(self) -> Path:
        return self.repo_root / "outputs"

    @property
    def midi_examples(self) -> Path:
        return self.outputs / "midi_examples"

    @property
    def logs(self) -> Path:
        return self.repo_root / "logs"

    # File names
    dataset_file: str = "lpd5_piano_genre.npz"
    meta_file: str = "metadata.json"
    best_ckpt: str = "checkpoint_best.pt"
    latest_ckpt: str = "checkpoint_latest.pt"

    def make_dirs(self):
        for p in [self.data_processed, self.checkpoints, self.plots,
                  self.outputs, self.midi_examples, self.logs]:
            Path(p).mkdir(parents=True, exist_ok=True)


# Singleton instances — import these elsewhere
DATA   = DataConfig()
MODEL  = ModelConfig()
TRAIN  = TrainConfig()
INFER  = InferConfig()
PATHS  = PathConfig()
