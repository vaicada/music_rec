"""
Build FAISS index for Audio Autoencoder (Model 2) using tracks_features.csv data.

Loads the trained AudioAutoencoder, encodes ALL songs from the 1.2M dataset into
32-dimensional L2-normalised embeddings, builds a FAISS IndexFlatIP, and saves:
  - models/tracks_faiss.index
  - models/tracks_faiss.index.mappings.pkl

Usage:
    cd music_recommender
    python -m audio_model.build_audio_faiss
"""

import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_model.config2 import CONFIG
from audio_model.model2 import AudioAutoencoder


def load_and_normalize_npy(npy_path: str, stats_path: str) -> np.ndarray:
    """Load pre-saved .npy features and Z-score normalise."""
    features = np.load(npy_path).astype(np.float32)
    with open(stats_path) as f:
        stats = json.load(f)
    means = np.array(stats["mean"], dtype=np.float32)
    stds  = np.array(stats["std"],  dtype=np.float32)
    return (features - means) / (stds + 1e-8)


def build_index(
    model_path: str,
    train_features_path: str,
    train_meta_path: str,
    stats_path: str,
    faiss_out: str,
    mappings_out: str,
    latent_dim: int,
    batch_size: int = 4096,
    device_str: str = "cpu",
):
    """Encode all songs and build a FAISS IndexFlatIP from the latent embeddings."""

    device = torch.device(device_str)
    print(f"[build_faiss] Device: {device}")

    # ── Load Autoencoder model ────────────────────────────────────────────────
    print(f"[build_faiss] Loading model from {model_path} ...")
    model = AudioAutoencoder(
        input_dim=CONFIG.input_dim,
        latent_dim=latent_dim,
        dropout=0.0,        # No dropout during inference
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ── Load and normalise features ───────────────────────────────────────────
    print(f"[build_faiss] Loading features from {train_features_path} ...")
    features = load_and_normalize_npy(train_features_path, stats_path)
    print(f"[build_faiss] Dataset size: {len(features):,} songs")

    # ── Encode in batches ─────────────────────────────────────────────────────
    tensor_ds = TensorDataset(torch.from_numpy(features))
    loader    = DataLoader(tensor_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embeddings = []
    print("[build_faiss] Encoding embeddings ...")
    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="Encoding"):
            emb = model.encode(batch.to(device))
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"[build_faiss] Embeddings shape: {embeddings.shape}")

    # ── Build FAISS index ─────────────────────────────────────────────────────
    print("[build_faiss] Building FAISS IndexFlatIP ...")
    index = faiss.IndexFlatIP(latent_dim)   # Inner product == cosine for L2-normed vectors
    index.add(embeddings)  # type: ignore[call-arg]
    print(f"[build_faiss] FAISS index total vectors: {index.ntotal:,}")

    # ── Save index ────────────────────────────────────────────────────────────
    Path(faiss_out).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, faiss_out)
    print(f"[build_faiss] Saved FAISS index -> {faiss_out}")

    # ── Build and save metadata mappings ──────────────────────────────────────
    print(f"[build_faiss] Loading metadata from {train_meta_path} ...")
    meta_df = pd.read_parquet(train_meta_path)

    mappings = []
    for _, row in meta_df.iterrows():
        mappings.append({
            "song":   str(row.get("song_name", row.get("name", "Unknown"))),
            "artist": str(row.get("artist", "Unknown")),
            "year":   str(row.get("year", "")),
            "album":  str(row.get("album", "")),
            # No 'emotion' field – not available in tracks_features.csv
        })

    with open(mappings_out, "wb") as f:
        pickle.dump(mappings, f)
    print(f"[build_faiss] Saved mappings ({len(mappings):,} entries) -> {mappings_out}")
    print("[build_faiss] Done!")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    build_index(
        model_path=CONFIG.model_path,
        train_features_path=CONFIG.train_features_path,
        train_meta_path=CONFIG.train_meta_path,
        stats_path=CONFIG.stats_path,
        faiss_out=CONFIG.faiss_index_path,
        mappings_out=CONFIG.faiss_mappings_path,
        latent_dim=CONFIG.output_dim,
        batch_size=4096,
        device_str=device_str,
    )
