"""
Single entry point. Run: python train.py
Auto-resumes from latest checkpoint.
"""
import sys, torch
sys.path.append(".")

from configs.config import MODEL, TRAIN, PATHS
from src.dataset    import make_dataloaders
from src.unet       import UNet
from src.ddpm       import DDPM
from src.trainer    import Trainer


def main():
    PATHS.make_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    npz = PATHS.data_processed / PATHS.dataset_file
    if not npz.exists():
        raise FileNotFoundError(
            f"Dataset not found: {npz}\n"
            f"Run notebooks/01_data_pipeline.ipynb on Kaggle first."
        )

    train_loader, val_loader, genres = make_dataloaders(str(npz))

    unet  = UNet(MODEL).to(device)
    ddpm  = DDPM(unet, MODEL).to(device)
    optim = torch.optim.AdamW(unet.parameters(), lr=TRAIN.lr)

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"Parameters: {n_params:,}")

    trainer = Trainer(
        ddpm, optim, device,
        ckpt_dir=PATHS.checkpoints,
        log_path=PATHS.logs / "train_log.json",
    )
    trainer.fit(train_loader, val_loader, n_epochs=TRAIN.n_epochs)


if __name__ == "__main__":
    main()
