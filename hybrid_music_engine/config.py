"""
Configuration settings for the Hybrid Music Recommendation Engine.

This module contains all hyperparameters, paths, and settings for the model.

Author: Graduation Project
Created: 2026-01-06
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    # Project root directory
    project_root: str = str(Path(__file__).parent.parent)
    
    # Data paths
    data_dir: str = field(default="")
    dataset_dir: str = field(default="")  # Main folder for CSV/JSON datasets
    processed_data_dir: str = field(default="")
    
    # Model paths
    model_dir: str = field(default="")
    bert_cache_dir: str = field(default="")
    faiss_index_path: str = field(default="")
    
    # Log paths
    log_file: str = field(default="")
    
    # Dataset file paths
    spotify_dataset_path: str = field(default="")
    bert_dataset_path: str = field(default="")
    tracks_features_path: str = field(default="")
    
    def __post_init__(self):
        """Initialize paths after dataclass creation."""
        if not self.data_dir:
            self.data_dir = os.path.join(self.project_root, "data")
        if not self.dataset_dir:
            self.dataset_dir = os.path.join(self.project_root, "dataset")
        if not self.processed_data_dir:
            self.processed_data_dir = os.path.join(self.data_dir, "processed")
        if not self.model_dir:
            self.model_dir = os.path.join(self.project_root, "models")
        if not self.bert_cache_dir:
            self.bert_cache_dir = os.path.join(self.model_dir, "bert_cache")
        if not self.faiss_index_path:
            self.faiss_index_path = os.path.join(self.model_dir, "faiss_index.bin")
        if not self.log_file:
            self.log_file = os.path.join(self.project_root, "project_implementation_log.txt")
        
        # Dataset file paths
        if not self.spotify_dataset_path:
            self.spotify_dataset_path = os.path.join(self.dataset_dir, "spotify_dataset.csv")
        if not self.bert_dataset_path:
            self.bert_dataset_path = os.path.join(self.dataset_dir, "final_milliondataset_BERT_500K_revised.json")
        if not self.tracks_features_path:
            self.tracks_features_path = os.path.join(self.dataset_dir, "tracks_features.csv")
    
    def ensure_dirs_exist(self) -> None:
        """Create all directories if they don't exist."""
        for attr_name in ['data_dir', 'dataset_dir', 'processed_data_dir', 
                          'model_dir', 'bert_cache_dir']:
            path = getattr(self, attr_name)
            os.makedirs(path, exist_ok=True)



@dataclass
class BERTConfig:
    """Configuration for BERT model."""
    
    # Model selection
    model_name: str = "bert-base-uncased"
    
    # Tokenization
    max_length: int = 256  # Max tokens for lyrics
    padding: str = "max_length"
    truncation: bool = True
    
    # BERT output
    hidden_size: int = 768  # BERT base hidden size
    output_dim: int = 256   # Compressed output dimension
    
    # Fine-tuning settings
    freeze_bert_layers: int = 10  # Number of BERT layers to freeze (0-12)
    dropout: float = 0.3


@dataclass
class AudioConfig:
    """Configuration for Audio Features branch."""
    
    # Input features (9 features - removed 'mode' and 'loudness')
    # Note: 'loudness' was removed because it's 100% NaN in the processed dataset
    audio_features: List[str] = field(default_factory=lambda: [
        'energy',
        'danceability', 
        'valence',
        'tempo',
        'acousticness',
        'instrumentalness',
        'speechiness',
        'liveness',
        'key'
    ])
    
    # Network architecture
    input_dim: int = 9  # Number of audio features (was 10, removed 'loudness')
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 64])
    output_dim: int = 64
    dropout: float = 0.2
    
    # Normalization
    normalize_features: bool = True


@dataclass
class MetadataConfig:
    """Configuration for Metadata Embedding branch."""
    
    # Categorical features - Updated to match actual data
    genre_vocab_size: int = 3500    # Increased: 3010 unique genres in dataset
    key_vocab_size: int = 12        # Musical keys (C, C#, D, etc.)
    emotion_vocab_size: int = 20    # Increased: 13 unique emotions in dataset
    
    # Embedding dimensions
    genre_embedding_dim: int = 16
    key_embedding_dim: int = 8
    emotion_embedding_dim: int = 8
    
    # Output
    output_dim: int = 32
    dropout: float = 0.2


@dataclass
class FusionConfig:
    """Configuration for the Fusion layer."""
    
    # Input dimensions (sum of all branch outputs)
    bert_dim: int = 256
    audio_dim: int = 64
    metadata_dim: int = 32
    
    @property
    def total_input_dim(self) -> int:
        return self.bert_dim + self.audio_dim + self.metadata_dim
    
    # Hidden layers
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    
    # Final embedding
    final_embedding_dim: int = 64  # The 64D output vector
    
    # Regularization
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Optimizer
    optimizer: str = "AdamW"
    adam_epsilon: float = 1e-8
    
    # Loss weights for multi-task learning
    triplet_loss_weight: float = 0.5
    emotion_loss_weight: float = 0.3
    context_loss_weight: float = 0.2
    
    # Triplet loss margin
    triplet_margin: float = 0.3
    
    # Training behavior
    early_stopping_patience: int = 5
    save_best_only: bool = True
    gradient_clip_value: float = 1.0
    
    # Hardware - Auto-detect GPU
    device: str = "auto"  # "auto", "cuda", or "cpu"
    num_workers: int = 0  # Set to 0 for Windows compatibility
    pin_memory: bool = True
    
    def get_device(self) -> str:
        """Get the actual device to use (auto-detect if 'auto')."""
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
    
    # Validation
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class FAISSConfig:
    """Configuration for FAISS indexing."""
    
    # Index type - Accuracy > Speed, so use exact search
    index_type: str = "IndexFlatL2"  # Exact L2 search
    
    # Alternative for large datasets
    use_hnsw: bool = False
    hnsw_m: int = 64          # Number of connections per layer
    hnsw_ef_construction: int = 200  # Construction time parameter
    hnsw_ef_search: int = 128  # Search time parameter
    
    # Embedding dimension
    embedding_dim: int = 64
    
    # Search parameters
    default_top_k: int = 10
    
    # GPU settings - Enable GPU by default if available
    use_gpu: bool = True  # Auto-use GPU for FAISS if available
    gpu_id: int = 0


@dataclass
class Config:
    """Master configuration class combining all configs."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    bert: BERTConfig = field(default_factory=BERTConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    
    # Project info
    project_name: str = "Hybrid Music Recommender"
    version: str = "1.0.0"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "project_name": self.project_name,
            "version": self.version,
            "bert_model": self.bert.model_name,
            "bert_max_length": self.bert.max_length,
            "audio_features": self.audio.audio_features,
            "final_embedding_dim": self.fusion.final_embedding_dim,
            "batch_size": self.training.batch_size,
            "epochs": self.training.epochs,
            "learning_rate": self.training.learning_rate,
            "faiss_index_type": self.faiss.index_type,
        }
    
    def validate(self) -> bool:
        """Validate configuration consistency."""
        # Check embedding dimensions match
        assert self.fusion.bert_dim == self.bert.output_dim, \
            "BERT output dim must match fusion input"
        assert self.fusion.audio_dim == self.audio.output_dim, \
            "Audio output dim must match fusion input"
        assert self.fusion.metadata_dim == self.metadata.output_dim, \
            "Metadata output dim must match fusion input"
        assert self.faiss.embedding_dim == self.fusion.final_embedding_dim, \
            "FAISS dim must match final embedding dim"
        return True


# Default configuration instance
DEFAULT_CONFIG = Config()


def get_config(custom_config: Optional[dict] = None) -> Config:
    """
    Get configuration, optionally with custom overrides.
    
    Args:
        custom_config: Dictionary of custom configuration values.
    
    Returns:
        Config instance.
    """
    config = Config()
    
    if custom_config:
        # Apply custom overrides (simplified - can be extended)
        if "batch_size" in custom_config:
            config.training.batch_size = custom_config["batch_size"]
        if "epochs" in custom_config:
            config.training.epochs = custom_config["epochs"]
        if "learning_rate" in custom_config:
            config.training.learning_rate = custom_config["learning_rate"]
        if "device" in custom_config:
            config.training.device = custom_config["device"]
    
    config.validate()
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Project: {config.project_name} v{config.version}")
    print(f"BERT Model: {config.bert.model_name}")
    print(f"Final Embedding Dim: {config.fusion.final_embedding_dim}D")
    print(f"Training Epochs: {config.training.epochs}")
    print(f"FAISS Index Type: {config.faiss.index_type}")
    print("\nFull config dict:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")
