"""
Hybrid Music Recommendation Engine

A modular Python package for music recommendations using a hybrid multi-modal
architecture combining BERT (lyrics), audio features, and metadata.

Architecture:
    - Branch A (NLP): BERT encoder for lyrics and emotion understanding
    - Branch B (Audio): MLP encoder for Spotify audio features
    - Branch C (Metadata): Embedding layers for genre, key, and emotion
    - Fusion Layer: Combines all branches into 64D embeddings
    - FAISS Index: Fast similarity search for recommendations

Usage:
    >>> from hybrid_music_engine import MusicRecommendationEngine
    >>> engine = MusicRecommendationEngine()
    >>> engine.load_model("path/to/model.pth")
    >>> engine.load_index("path/to/index.bin")
    >>> engine.load_song_data("path/to/songs.csv")
    >>> recommendations = engine.get_similar_songs("Bohemian Rhapsody", "Queen", top_k=5)

Author: Graduation Project
Created: 2026-01-06
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Graduation Project"

# Core components
from .config import (
    Config,
    get_config,
    PathConfig,
    BERTConfig,
    AudioConfig,
    MetadataConfig,
    FusionConfig,
    TrainingConfig,
    FAISSConfig,
    DEFAULT_CONFIG
)

from .logger import (
    StepLogger,
    get_logger,
    log_step
)

from .processors import (
    TextProcessor,
    AudioProcessor,
    MetadataProcessor,
    MusicDataset,
    DataManager
)

from .model import (
    HybridMusicModel
)

# from .trainer import (
#     TripletMiner,
#     Trainer
# )

from .inference import (
    FAISSIndex,
    MusicRecommendationEngine,
    create_engine
)


# Convenience aliases
Engine = MusicRecommendationEngine
Model = HybridMusicModel
Logger = StepLogger


# Package-level initialization
def init_logger(log_file: str = "project_implementation_log.txt") -> StepLogger:
    """Initialize the global logger."""
    return get_logger(log_file)


def quick_start(
    model_path: str,
    index_path: str,
    data_path: str
) -> MusicRecommendationEngine:
    """
    Quick start function to create and initialize the engine.
    
    Args:
        model_path: Path to trained model checkpoint.
        index_path: Path to FAISS index file.
        data_path: Path to song database file.
    
    Returns:
        Initialized MusicRecommendationEngine ready for recommendations.
    
    Example:
        >>> from hybrid_music_engine import quick_start
        >>> engine = quick_start("model.pth", "index.bin", "songs.csv")
        >>> results = engine.get_similar_songs("Hello", "Adele")
    """
    return create_engine(model_path, index_path, data_path)


__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Configuration
    "Config",
    "get_config",
    "PathConfig",
    "BERTConfig",
    "AudioConfig",
    "MetadataConfig",
    "FusionConfig",
    "TrainingConfig",
    "FAISSConfig",
    "DEFAULT_CONFIG",
    
    # Logging
    "StepLogger",
    "get_logger",
    "log_step",
    "Logger",
    
    # Data Processing
    "TextProcessor",
    "AudioProcessor",
    "MetadataProcessor",
    "MusicDataset",
    "DataManager",
    
    # Model
    "HybridMusicModel",
    "Model",
    
    # Training
    # "TripletMiner",
    # "Trainer",
    
    # Inference
    "FAISSIndex",
    "MusicRecommendationEngine",
    "create_engine",
    "Engine",
    
    # Convenience
    "init_logger",
    "quick_start"
]
