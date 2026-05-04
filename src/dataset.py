"""
Piano roll dataset for PyTorch.
Loads the preprocessed .npz, returns (roll, genre) tuples.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from configs.config import DATA, TRAIN


class PianoRollDataset(Dataset):
    """
    Each item: ((1, 64, 88) tensor, int label)
    Augmentation: time shift + pitch transpose (zero-padded, non-circular).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        # Add channel dim: (N, 64, 88) -> (N, 1, 64, 88)
        # Scale to [-1, 1] — standard for diffusion
        X_scaled = X * 2.0 - 1.0
        self.X = torch.from_numpy(X_scaled).unsqueeze(1).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        roll  = self.X[idx].clone()
        label = self.y[idx]
        if self.augment:
            roll = self._augment(roll)
        return roll, label

    @staticmethod
    def _augment(roll: torch.Tensor) -> torch.Tensor:
        """Time shift ±8 steps, pitch transpose ±2 semitones (zero-padded, non-circular).

        Uses torch.roll followed by zeroing the vacated boundary so that no values
        wrap from one end to the other — musically invalid wrap-around is avoided.
        Silence is represented as -1 in the [-1, 1] scaled space.
        """
        time_shift  = torch.randint(-8, 9, (1,)).item()
        pitch_shift = torch.randint(-2, 3, (1,)).item()

        # Time shift along dim=1; fill vacated boundary with -1 (silence)
        if time_shift != 0:
            roll = torch.roll(roll, shifts=time_shift, dims=1)
            if time_shift > 0:
                roll[:, :time_shift, :] = -1.0
            else:
                roll[:, time_shift:, :] = -1.0

        # Pitch shift along dim=2; fill vacated boundary with -1 (silence)
        if pitch_shift != 0:
            roll = torch.roll(roll, shifts=pitch_shift, dims=2)
            if pitch_shift > 0:
                roll[:, :, :pitch_shift] = -1.0
            else:
                roll[:, :, pitch_shift:] = -1.0

        return roll


def make_dataloaders(npz_path: str, batch_size: int = TRAIN.batch_size):
    """Returns: train_loader, val_loader, genres_list"""
    data    = np.load(npz_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    genres  = list(data["genres"])

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Genres: {genres}")
    print(f"Sparsity: {(X_train > 0).mean():.3%}")

    train_ds = PianoRollDataset(X_train, y_train, augment=True)
    val_ds   = PianoRollDataset(X_val,   y_val,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=TRAIN.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=TRAIN.num_workers, pin_memory=True
    )
    return train_loader, val_loader, genres
