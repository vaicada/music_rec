"""
Data Processors for the Hybrid Music Recommendation Engine.

This module handles data loading, cleaning, tokenization, and preprocessing
for all three branches (BERT, Audio, Metadata).

Author: Graduation Project
Created: 2026-01-06
"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertTokenizerFast

from .logger import get_logger, log_step
from .config import Config, get_config


class TextProcessor:
    """
    Text processing for lyrics and emotion tags.
    
    Handles cleaning, tokenization, and BERT encoding of text data.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the TextProcessor.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or get_config()
        self.logger = get_logger(self.config.paths.log_file)
        self._tokenizer = None
    
    @property
    def tokenizer(self) -> BertTokenizerFast:
        """Lazy loading of BERT tokenizer."""
        if self._tokenizer is None:
            self.logger.log_start("Loading BERT Tokenizer", "MODEL")
            self._tokenizer = BertTokenizerFast.from_pretrained(
                self.config.bert.model_name,
                cache_dir=self.config.paths.bert_cache_dir
            )
            self.logger.log_end("Loading BERT Tokenizer", "MODEL")
        return self._tokenizer
    
    def clean_lyrics(self, text: str) -> str:
        """
        Clean and normalize lyrics text.
        
        Args:
            text: Raw lyrics text.
        
        Returns:
            Cleaned text string.
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove section markers like [Verse 1], [Chorus], etc.
        text = re.sub(r'\[.*?\]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\'\"\.\,\!\?]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def combine_text_features(
        self, 
        lyrics: str, 
        emotion: str = "", 
        genre: str = ""
    ) -> str:
        """
        Combine lyrics with emotion and genre for richer context.
        
        Args:
            lyrics: Song lyrics.
            emotion: Emotion label.
            genre: Genre label.
        
        Returns:
            Combined text string.
        """
        parts = []
        
        if emotion:
            parts.append(f"[EMOTION: {emotion}]")
        if genre:
            parts.append(f"[GENRE: {genre}]")
        if lyrics:
            parts.append(lyrics)
        
        return " ".join(parts)
    
    @log_step("DATA", "Tokenizing Text Batch")
    def tokenize_batch(
        self, 
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts for BERT.
        
        Args:
            texts: List of text strings.
        
        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids.
        """
        encoded = self.tokenizer(
            texts,
            max_length=self.config.bert.max_length,
            padding=self.config.bert.padding,
            truncation=self.config.bert.truncation,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded.get("token_type_ids", 
                                          torch.zeros_like(encoded["input_ids"]))
        }
    
    def tokenize_single(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text string."""
        return self.tokenize_batch([text])


class AudioProcessor:
    """
    Audio features processing and normalization.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the AudioProcessor.
        
        Args:
            config: Configuration object.
        """
        self.config = config or get_config()
        self.logger = get_logger(self.config.paths.log_file)
        
        # Normalization statistics
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of audio feature names."""
        return self.config.audio.audio_features
    
    @log_step("DATA", "Computing Normalization Statistics")
    def fit(self, data: pd.DataFrame) -> 'AudioProcessor':
        """
        Compute normalization statistics from training data.
        
        Args:
            data: DataFrame with audio features.
        
        Returns:
            Self for chaining.
        """
        features = data[self.feature_names].values.astype(np.float32)
        
        self._mean = np.mean(features, axis=0)
        self._std = np.std(features, axis=0) + 1e-8  # Prevent division by zero
        self._min = np.min(features, axis=0)
        self._max = np.max(features, axis=0)
        
        self.logger.log(
            "Normalization statistics computed",
            "DATA",
            details={
                "num_features": len(self.feature_names),
                "samples": len(data)
            }
        )
        
        return self
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> torch.Tensor:
        """
        Normalize audio features.
        
        Args:
            data: DataFrame or array with audio features.
        
        Returns:
            Normalized tensor of shape (batch_size, num_features).
        """
        if self._mean is None:
            raise ValueError("AudioProcessor not fitted. Call fit() first.")
        
        if isinstance(data, pd.DataFrame):
            features = data[self.feature_names].values.astype(np.float32)
        else:
            features = np.array(data, dtype=np.float32)
        
        # Z-score normalization
        normalized = (features - self._mean) / self._std
        
        return torch.tensor(normalized, dtype=torch.float32)
    
    def fit_transform(self, data: pd.DataFrame) -> torch.Tensor:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    def save_stats(self, path: str) -> None:
        """Save normalization statistics to file."""
        stats = {
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "features": self.feature_names
        }
        with open(path, 'w') as f:
            json.dump(stats, f)
        self.logger.log(f"Saved audio stats to {path}", "DATA")
    
    def load_stats(self, path: str) -> None:
        """Load normalization statistics from file."""
        with open(path, 'r') as f:
            stats = json.load(f)
        self._mean = np.array(stats["mean"], dtype=np.float32)
        self._std = np.array(stats["std"], dtype=np.float32)
        self._min = np.array(stats["min"], dtype=np.float32)
        self._max = np.array(stats["max"], dtype=np.float32)
        self.logger.log(f"Loaded audio stats from {path}", "DATA")


class MetadataProcessor:
    """
    Metadata processing for categorical features.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the MetadataProcessor.
        
        Args:
            config: Configuration object.
        """
        self.config = config or get_config()
        self.logger = get_logger(self.config.paths.log_file)
        
        # Vocabulary mappings
        self.genre_to_idx: Dict[str, int] = {}
        self.key_to_idx: Dict[str, int] = {}
        self.emotion_to_idx: Dict[str, int] = {}
        
        # Initialize musical keys
        self._init_key_mapping()
    
    def _init_key_mapping(self) -> None:
        """Initialize musical key mappings."""
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.key_to_idx = {k: i for i, k in enumerate(keys)}
        # Also map numeric keys
        for i in range(12):
            self.key_to_idx[str(i)] = i
            self.key_to_idx[i] = i
    
    @log_step("DATA", "Building Vocabulary")
    def fit(self, data: pd.DataFrame) -> 'MetadataProcessor':
        """
        Build vocabulary from training data.
        
        Args:
            data: DataFrame with metadata columns.
        
        Returns:
            Self for chaining.
        """
        # Build genre vocabulary
        if 'Genre' in data.columns or 'genre' in data.columns:
            genre_col = 'Genre' if 'Genre' in data.columns else 'genre'
            genres = data[genre_col].dropna().unique()
            self.genre_to_idx = {g: i for i, g in enumerate(genres)}
            self.genre_to_idx['<UNK>'] = len(self.genre_to_idx)
        
        # Build emotion vocabulary
        if 'emotion' in data.columns:
            emotions = data['emotion'].dropna().unique()
            self.emotion_to_idx = {e: i for i, e in enumerate(emotions)}
            self.emotion_to_idx['<UNK>'] = len(self.emotion_to_idx)
        
        self.logger.log(
            "Vocabulary built",
            "DATA",
            details={
                "genres": len(self.genre_to_idx),
                "emotions": len(self.emotion_to_idx),
                "keys": len(self.key_to_idx)
            }
        )
        
        return self
    
    def transform_genre(self, genres: List[str]) -> torch.Tensor:
        """Convert genre strings to indices."""
        unk_idx = self.genre_to_idx.get('<UNK>', 0)
        indices = [self.genre_to_idx.get(g, unk_idx) for g in genres]
        return torch.tensor(indices, dtype=torch.long)
    
    def transform_key(self, keys: List[Union[str, int]]) -> torch.Tensor:
        """Convert key values to indices."""
        indices = [self.key_to_idx.get(k, 0) for k in keys]
        return torch.tensor(indices, dtype=torch.long)
    
    def transform_emotion(self, emotions: List[str]) -> torch.Tensor:
        """Convert emotion strings to indices."""
        unk_idx = self.emotion_to_idx.get('<UNK>', 0)
        indices = [self.emotion_to_idx.get(e, unk_idx) for e in emotions]
        return torch.tensor(indices, dtype=torch.long)
    
    def save_vocab(self, path: str) -> None:
        """Save vocabularies to file."""
        vocab = {
            "genre_to_idx": self.genre_to_idx,
            "key_to_idx": {str(k): v for k, v in self.key_to_idx.items()},
            "emotion_to_idx": self.emotion_to_idx
        }
        with open(path, 'w') as f:
            json.dump(vocab, f)
        self.logger.log(f"Saved vocabulary to {path}", "DATA")
    
    def load_vocab(self, path: str) -> None:
        """Load vocabularies from file."""
        with open(path, 'r') as f:
            vocab = json.load(f)
        self.genre_to_idx = vocab["genre_to_idx"]
        self.emotion_to_idx = vocab["emotion_to_idx"]
        self.logger.log(f"Loaded vocabulary from {path}", "DATA")


class MusicDataset(Dataset):
    """
    PyTorch Dataset for music data.
    
    Handles loading and preprocessing of all three data types:
    lyrics (text), audio features, and metadata.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        text_processor: TextProcessor,
        audio_processor: AudioProcessor,
        metadata_processor: MetadataProcessor,
        config: Optional[Config] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with all features.
            text_processor: TextProcessor instance.
            audio_processor: AudioProcessor instance (fitted).
            metadata_processor: MetadataProcessor instance (fitted).
            config: Configuration object.
        """
        self.config = config or get_config()
        self.data = data.reset_index(drop=True)
        self.text_processor = text_processor
        self.audio_processor = audio_processor
        self.metadata_processor = metadata_processor
        
        # Get column names
        self.lyrics_col = 'text' if 'text' in data.columns else 'lyrics'
        self.genre_col = 'Genre' if 'Genre' in data.columns else 'genre'
        self.emotion_col = 'emotion'
        self.key_col = 'key' if 'key' in data.columns else 'Key'
        
        # Pre-process audio features
        self.audio_features = audio_processor.transform(data)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with all processed features.
        """
        row = self.data.iloc[idx]
        
        # Process text (lyrics + emotion + genre)
        lyrics = self.text_processor.clean_lyrics(row.get(self.lyrics_col, ""))
        emotion = str(row.get(self.emotion_col, ""))
        genre = str(row.get(self.genre_col, ""))
        
        combined_text = self.text_processor.combine_text_features(
            lyrics, emotion, genre
        )
        text_encoded = self.text_processor.tokenize_single(combined_text)
        
        # Audio features (already processed)
        audio = self.audio_features[idx]
        
        # Metadata
        genre_idx = self.metadata_processor.transform_genre([genre])[0]
        key_idx = self.metadata_processor.transform_key([row.get(self.key_col, 0)])[0]
        emotion_idx = self.metadata_processor.transform_emotion([emotion])[0]
        
        return {
            "input_ids": text_encoded["input_ids"].squeeze(0),
            "attention_mask": text_encoded["attention_mask"].squeeze(0),
            "token_type_ids": text_encoded["token_type_ids"].squeeze(0),
            "audio_features": audio,
            "genre_idx": genre_idx,
            "key_idx": key_idx,
            "emotion_idx": emotion_idx,
            "emotion_label": emotion_idx,  # For supervised training
            "idx": torch.tensor(idx, dtype=torch.long)
        }


class DataManager:
    """
    High-level data manager for the music recommendation system.
    
    Handles loading, preprocessing, and creating data loaders.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the DataManager.
        
        Args:
            config: Configuration object.
        """
        self.config = config or get_config()
        self.logger = get_logger(self.config.paths.log_file)
        
        # Initialize processors
        self.text_processor = TextProcessor(config)
        self.audio_processor = AudioProcessor(config)
        self.metadata_processor = MetadataProcessor(config)
        
        # Data storage
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
    
    @log_step("DATA", "Loading Dataset")
    def load_data(
        self, 
        path: str, 
        file_format: str = "auto"
    ) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            path: Path to data file.
            file_format: 'csv', 'json', 'jsonl', or 'auto'.
        
        Returns:
            Loaded DataFrame.
        """
        path = Path(path)
        
        if file_format == "auto":
            file_format = path.suffix.lower().strip('.')
        
        if file_format == "csv":
            data = pd.read_csv(path)
        elif file_format == "json":
            data = pd.read_json(path)
        elif file_format in ["jsonl", "jl"]:
            data = pd.read_json(path, lines=True)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        self.logger.log(
            f"Loaded data from {path.name}",
            "DATA",
            details={"rows": len(data), "columns": len(data.columns)}
        )
        
        return data
    
    @log_step("DATA", "Preprocessing Data")
    def preprocess(
        self, 
        data: pd.DataFrame,
        fit_processors: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess the data.
        
        Args:
            data: Raw DataFrame.
            fit_processors: Whether to fit processors on this data.
        
        Returns:
            Preprocessed DataFrame.
        """
        # Clean lyrics
        lyrics_col = 'text' if 'text' in data.columns else 'lyrics'
        if lyrics_col in data.columns:
            data[lyrics_col] = data[lyrics_col].apply(
                self.text_processor.clean_lyrics
            )
        
        # Handle missing values in audio features
        for feat in self.config.audio.audio_features:
            if feat in data.columns:
                data[feat] = data[feat].fillna(data[feat].median())
        
        # Fit processors if needed
        if fit_processors:
            self.audio_processor.fit(data)
            self.metadata_processor.fit(data)
        
        return data
    
    @log_step("DATA", "Splitting Data")
    def split_data(
        self, 
        data: pd.DataFrame,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Full DataFrame.
            val_size: Validation set proportion.
            test_size: Test set proportion.
            random_state: Random seed.
        
        Returns:
            Tuple of (train, val, test) DataFrames.
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state
        )
        
        self.train_data = train.reset_index(drop=True)
        self.val_data = val.reset_index(drop=True)
        self.test_data = test.reset_index(drop=True)
        
        self.logger.log(
            "Data split completed",
            "DATA",
            details={
                "train": len(train),
                "val": len(val),
                "test": len(test)
            }
        )
        
        return self.train_data, self.val_data, self.test_data
    
    def create_dataset(self, data: pd.DataFrame) -> MusicDataset:
        """Create a PyTorch Dataset from DataFrame."""
        return MusicDataset(
            data=data,
            text_processor=self.text_processor,
            audio_processor=self.audio_processor,
            metadata_processor=self.metadata_processor,
            config=self.config
        )
    
    def create_dataloader(
        self, 
        data: pd.DataFrame,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        num_workers: Optional[int] = None
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader.
        
        Args:
            data: DataFrame to load.
            batch_size: Batch size (uses config default if not specified).
            shuffle: Whether to shuffle data.
            num_workers: Number of worker processes.
        
        Returns:
            DataLoader instance.
        """
        dataset = self.create_dataset(data)
        
        return DataLoader(
            dataset,
            batch_size=batch_size or self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Force 0 for Windows - logger has unpicklable _file_lock
            pin_memory=False  # Disable pin_memory when no GPU accelerator detected
        )


if __name__ == "__main__":
    # Test the processors
    print("Testing processors...")
    
    config = get_config()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'text': ["Hello world, this is a song", "Another song lyrics here"],
        'energy': [0.8, 0.6],
        'danceability': [0.7, 0.5],
        'valence': [0.9, 0.3],
        'tempo': [120.0, 90.0],
        'acousticness': [0.1, 0.8],
        'instrumentalness': [0.0, 0.2],
        'speechiness': [0.05, 0.1],
        'liveness': [0.1, 0.2],
        'loudness': [-5.0, -10.0],
        'key': [0, 5],
        'mode': [1, 0],
        'Genre': ['pop', 'rock'],
        'emotion': ['joy', 'sadness']
    })
    
    # Test TextProcessor
    text_proc = TextProcessor(config)
    cleaned = text_proc.clean_lyrics("  Hello   [Verse 1] World!  ")
    print(f"Cleaned text: '{cleaned}'")
    
    # Test AudioProcessor
    audio_proc = AudioProcessor(config)
    audio_proc.fit(sample_data)
    audio_tensor = audio_proc.transform(sample_data)
    print(f"Audio tensor shape: {audio_tensor.shape}")
    
    # Test MetadataProcessor
    meta_proc = MetadataProcessor(config)
    meta_proc.fit(sample_data)
    genre_indices = meta_proc.transform_genre(['pop', 'rock'])
    print(f"Genre indices: {genre_indices}")
    
    print("All processors working correctly!")
