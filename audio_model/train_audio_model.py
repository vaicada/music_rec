"""
Train the Audio Autoencoder (Model 2) on tracks_features.csv data.

Loss function: MSELoss (reconstruction of 9 audio features)
No emotion labels required – fully Unsupervised Learning.

Usage:
    cd music_recommender
    python -m audio_model.train_audio_model
    python -m audio_model.train_audio_model --quick-test
    python -m audio_model.train_audio_model --epochs 20
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_model.config2 import CONFIG
from audio_model.model2 import AudioAutoencoder


# ── Dataset helper ────────────────────────────────────────────────────────────

def load_features(npy_path: str, stats_path: str) -> torch.Tensor:
    """Load pre-saved .npy features and Z-score normalise with pre-computed stats."""
    features = np.load(npy_path).astype(np.float32)   # shape: (N, 9)

    with open(stats_path) as f:
        stats = json.load(f)

    means = np.array(stats["mean"], dtype=np.float32)
    stds  = np.array(stats["std"],  dtype=np.float32)
    features = (features - means) / (stds + 1e-8)

    return torch.from_numpy(features)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_n = 0
    for (x,) in tqdm(loader, desc="  training", leave=False):
        x = x.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out["reconstruction"], x)   # MSE: recon vs original
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / total_n


@torch.no_grad()
def eval_epoch(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    for (x,) in loader:
        x = x.to(device)
        out = model(x)
        loss = criterion(out["reconstruction"], x)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / total_n


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true",
                        help="Use only first 50K rows and 3 epochs for a smoke test")
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("[train] Loading features...")
    train_feats = load_features(CONFIG.train_features_path, CONFIG.stats_path)
    val_feats   = load_features(CONFIG.val_features_path,   CONFIG.stats_path)

    if args.quick_test:
        print("[train] Quick-test: slicing to 50K train / 10K val")
        train_feats = train_feats[:50_000]
        val_feats   = val_feats[:10_000]
        args.epochs = min(args.epochs, 3)

    print(f"[train] Train: {len(train_feats)} | Val: {len(val_feats)}")

    train_loader = DataLoader(
        TensorDataset(train_feats),
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(val_feats),
        batch_size=CONFIG.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AudioAutoencoder(
        input_dim=CONFIG.input_dim,
        latent_dim=CONFIG.output_dim,
        dropout=CONFIG.dropout,
    ).to(device)

    print(f"[train] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion  = nn.MSELoss()

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss   = float("inf")
    best_state      = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"[train] Starting training for up to {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(train_loader, model, optimizer, criterion, device)
        val_loss   = eval_epoch(val_loader,   model, criterion, device)
        scheduler.step()

        print(f"  Epoch {epoch:03d}/{args.epochs} | "
              f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  -> New best model (Val MSE: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG.early_stopping_patience:
                print(f"[train] Early stopping after {epoch} epochs.")
                break

    # ── Save best model ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    Path(CONFIG.model_dir).mkdir(parents=True, exist_ok=True)
    torch.save(best_state, CONFIG.model_path)
    print(f"[train] Saved best model -> {CONFIG.model_path}")
    print(f"[train] Best Validation MSE: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
