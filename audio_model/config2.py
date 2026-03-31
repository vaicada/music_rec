import os
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class AudioOnlyConfig:
    project_root: str = str(Path(__file__).parent.parent)
    model_dir: str = os.path.join(project_root, "models")
    data_dir: str = os.path.join(project_root, "data", "processed", "tracks")

    # Paths – now pointing at .npy / .parquet files for 1.2M track dataset
    train_features_path: str = os.path.join(data_dir, "features_train.npy")
    val_features_path: str = os.path.join(data_dir, "features_val.npy")
    test_features_path: str = os.path.join(data_dir, "features_test.npy")
    train_meta_path: str = os.path.join(data_dir, "meta_train.parquet")
    val_meta_path: str = os.path.join(data_dir, "meta_val.parquet")

    # Normalization stats computed from train set
    stats_path: str = os.path.join(model_dir, "tracks_stats.json")

    # Model artifacts
    model_path: str = os.path.join(model_dir, "autoencoder_model2.pth")
    faiss_index_path: str = os.path.join(model_dir, "tracks_faiss.index")
    faiss_mappings_path: str = os.path.join(model_dir, "tracks_faiss.index.mappings.pkl")

    # Audio features (9 features – same as before, NO emotion label needed)
    audio_features: list = field(default_factory=lambda: [
        'energy', 'danceability', 'valence', 'tempo',
        'acousticness', 'instrumentalness', 'speechiness', 'liveness', 'key'
    ])

    # Model architecture – Autoencoder: 9 -> 16 -> 32 (bottleneck) -> 16 -> 9
    input_dim: int = 9
    output_dim: int = 32      # Bottleneck / Latent space dimension (same as before)
    dropout: float = 0.1

    # Training  (large dataset -> large batch / fewer epochs)
    batch_size: int = 2048
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5

CONFIG = AudioOnlyConfig()
