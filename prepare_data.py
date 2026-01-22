"""
Data Preparation Pipeline for Hybrid Music Recommendation Engine.

This script handles:
1.1 Data Loading & Exploration
    - Analyze all datasets
    - Merge datasets intelligently
    - Handle missing values
    - Create train/val/test splits (80/10/10)

1.2 Lyrics Processing
    - Clean lyrics text (remove URLs, special chars)
    - Tokenization using BERT tokenizer
    - Create lyrics embeddings using pre-trained BERT
    - Save embeddings to disk

Author: Graduation Project
Created: 2026-01-10
"""

import os
import sys
import json
import pickle
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

class DataPrepConfig:
    """Configuration for data preparation."""
    
    # Dataset paths
    DATASET_DIR = PROJECT_ROOT / "dataset"
    PRIMARY_DATASET = "final_milliondataset_BERT_500K_revised.json"  # 551K songs
    SECONDARY_DATASET = "900k Definitive Spotify Dataset.json"  # 900K songs (498K unique)
    SPOTIFY_CSV = "spotify_dataset.csv"  # 551K songs (CSV version of BERT 500K)
    TRACKS_FEATURES = "tracks_features.csv"  # 1.2M audio features only (no lyrics)
    
    # Output paths
    OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
    EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
    
    # BERT settings
    BERT_MODEL = "bert-base-uncased"
    MAX_LENGTH = 256
    EMBEDDING_DIM = 768  # BERT base hidden size
    
    # Processing settings
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    RANDOM_SEED = 42
    
    # Audio features to use (with possible alternate names)
    AUDIO_FEATURES = [
        'energy', 'danceability', 'valence', 'tempo',
        'acousticness', 'instrumentalness', 'speechiness',
        'liveness', 'loudness', 'key', 'mode'
    ]
    
    # Column name mapping (dataset column -> standard name)
    COLUMN_MAPPING = {
        'Energy': 'energy',
        'Danceability': 'danceability',
        'Positiveness': 'valence',  # Positiveness = Valence in Spotify
        'Tempo': 'tempo',
        'Acousticness': 'acousticness',
        'Instrumentalness': 'instrumentalness',
        'Speechiness': 'speechiness',
        'Liveness': 'liveness',
        'Loudness (db)': 'loudness',
        'Key': 'key',
        'Mode': 'mode',
        'Artist(s)': 'artist',
        'song': 'song_name',
        'text': 'lyrics',
        'Genre': 'genre',
        'Album': 'album',
        'Popularity': 'popularity'
    }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to standard format.
    Also converts audio feature values from strings to numeric.
    """
    df = df.copy()
    
    # Rename columns using mapping
    rename_map = {}
    for old_name, new_name in DataPrepConfig.COLUMN_MAPPING.items():
        if old_name in df.columns and new_name not in df.columns:
            rename_map[old_name] = new_name
    
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Renamed {len(rename_map)} columns: {list(rename_map.keys())}")
    
    # Convert audio features to numeric
    for feat in DataPrepConfig.AUDIO_FEATURES:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
    
    return df


def normalize_to_spotify_api(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize audio features to Spotify API standard (0-1 scale).
    
    PRIMARY dataset uses 0-100 scale for most features.
    Spotify API standard uses 0-1 scale.
    
    Also handles:
    - Loudness: Remove "db" suffix, keep as dB (-60 to 0)
    - Tempo: Keep as BPM (no normalization needed)
    - Key: Keep as 0-11 (no normalization needed)
    - Mode: Keep as 0/1 (no normalization needed)
    """
    df = df.copy()
    
    print("\n" + "=" * 60)
    print("NORMALIZING TO SPOTIFY API STANDARD (0-1 scale)")
    print("=" * 60)
    
    # Features that need 0-100 -> 0-1 conversion
    scale_features = ['energy', 'danceability', 'valence', 'acousticness', 
                      'instrumentalness', 'speechiness', 'liveness']
    
    # Features to keep as-is
    keep_as_is = ['tempo', 'key', 'mode', 'loudness']
    
    for feat in scale_features:
        if feat in df.columns:
            # First convert to numeric
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
            
            # Check if values are in 0-100 range (mean > 1 suggests 0-100 scale)
            mean_val = df[feat].mean()
            max_val = df[feat].max()
            
            if max_val > 1:  # Likely 0-100 scale
                df[feat] = df[feat] / 100.0
                print(f"  {feat}: Converted from 0-100 to 0-1 (was mean={mean_val:.1f}, max={max_val:.1f})")
            else:
                print(f"  {feat}: Already in 0-1 scale (mean={mean_val:.3f})")
    
    # Handle loudness specially - may have "db" suffix
    if 'loudness' in df.columns:
        # If string, remove "db" suffix
        if df['loudness'].dtype == object:
            df['loudness'] = df['loudness'].astype(str).str.replace('db', '', case=False)
            df['loudness'] = df['loudness'].str.replace('DB', '')
        df['loudness'] = pd.to_numeric(df['loudness'], errors='coerce')
        print(f"  loudness: Parsed to float (mean={df['loudness'].mean():.2f} dB)")
    
    # Handle tempo - may be string
    if 'tempo' in df.columns:
        df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')
        print(f"  tempo: Converted to numeric (mean={df['tempo'].mean():.1f} BPM)")
    
    # Handle key - may be string like "C", "D", etc. or numeric 0-11
    if 'key' in df.columns:
        # Try numeric first
        df['key'] = pd.to_numeric(df['key'], errors='coerce')
        # Key should be 0-11, if NaN fill with 0
        df['key'] = df['key'].fillna(0).astype(int) % 12
        print(f"  key: Normalized to 0-11 range")
    
    # Handle mode - should be 0 or 1
    if 'mode' in df.columns:
        df['mode'] = pd.to_numeric(df['mode'], errors='coerce')
        df['mode'] = df['mode'].fillna(0).astype(int) % 2
    
    # Verify all features are in correct range
    print("\n  Verification:")
    for feat in scale_features:
        if feat in df.columns:
            valid_data = df[feat].dropna()
            if len(valid_data) > 0:
                min_v, max_v, mean_v = valid_data.min(), valid_data.max(), valid_data.mean()
                status = "OK" if 0 <= min_v and max_v <= 1 else "WARN"
                print(f"    [{status}] {feat}: range [{min_v:.3f}, {max_v:.3f}], mean={mean_v:.3f}")
            else:
                print(f"    [SKIP] {feat}: all values are NaN")
    
    return df


# =============================================================================
# 1.1 DATA LOADING & EXPLORATION
# =============================================================================

def load_jsonl(filepath: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load JSONL file efficiently with progress bar."""
    print(f"Loading {filepath.name}...")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Reading JSONL")):
            if max_rows and i >= max_rows:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(data)
    print(f"  Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def load_csv(filepath: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load CSV file with progress indication."""
    print(f"Loading {filepath.name}...")
    
    nrows = max_rows if max_rows else None
    df = pd.read_csv(filepath, nrows=nrows, low_memory=False)
    
    print(f"  Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def analyze_dataset(df: pd.DataFrame, name: str) -> Dict:
    """Analyze a dataset and return statistics."""
    print(f"\n{'=' * 60}")
    print(f"ANALYZING: {name}")
    print(f"{'=' * 60}")
    
    stats = {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    print(f"Rows: {stats['rows']:,}")
    print(f"Columns: {stats['columns']}")
    print(f"\nColumn names:")
    for col in stats['column_names']:
        null_count = df[col].isnull().sum()
        null_pct = null_count / len(df) * 100
        print(f"  - {col}: {df[col].dtype} (null: {null_pct:.1f}%)")
    
    # Check for lyrics
    lyrics_cols = [c for c in df.columns if 'lyrics' in c.lower() or 'text' in c.lower()]
    stats['has_lyrics'] = len(lyrics_cols) > 0
    stats['lyrics_column'] = lyrics_cols[0] if lyrics_cols else None
    
    # Check for emotion
    emotion_cols = [c for c in df.columns if 'emotion' in c.lower()]
    stats['has_emotion'] = len(emotion_cols) > 0
    
    # Check for audio features
    audio_present = [f for f in DataPrepConfig.AUDIO_FEATURES if f in df.columns]
    stats['audio_features'] = audio_present
    stats['has_all_audio'] = len(audio_present) == len(DataPrepConfig.AUDIO_FEATURES)
    
    print(f"\nHas lyrics: {stats['has_lyrics']} ({stats['lyrics_column']})")
    print(f"Has emotion: {stats['has_emotion']}")
    print(f"Audio features: {len(audio_present)}/{len(DataPrepConfig.AUDIO_FEATURES)}")
    
    return stats


def load_and_merge_all_datasets(max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load and merge all 4 datasets intelligently.
    
    Strategy:
    1. Load primary (BERT 500K) - has lyrics, emotion, audio features, ISRC
    2. Load spotify_dataset.csv - check for additional columns
    3. Load secondary (900K Spotify) - add unique songs not in primary
    4. Load tracks_features.csv - enrich audio features for all songs
    
    Args:
        max_rows: Limit rows per dataset for testing
    
    Returns:
        Merged DataFrame with all unique songs
    """
    print("\n" + "=" * 60)
    print("LOADING ALL DATASETS")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. Load primary dataset (BERT 500K) - best quality, has ISRC
    # -------------------------------------------------------------------------
    primary_path = DataPrepConfig.DATASET_DIR / DataPrepConfig.PRIMARY_DATASET
    if primary_path.exists():
        primary_df = load_jsonl(primary_path, max_rows)
        primary_df = normalize_columns(primary_df)  # Standardize column names
        primary_df = normalize_to_spotify_api(primary_df)  # Convert 0-100 to 0-1
        print(f"  Primary (BERT 500K): {len(primary_df):,} songs (normalized)")
    else:
        print(f"  WARNING: Primary dataset not found: {primary_path}")
        primary_df = pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # 2. Load spotify_dataset.csv - may have additional columns
    # -------------------------------------------------------------------------
    spotify_csv_path = DataPrepConfig.DATASET_DIR / DataPrepConfig.SPOTIFY_CSV
    if spotify_csv_path.exists():
        spotify_csv_df = load_csv(spotify_csv_path, max_rows)
        print(f"  Spotify CSV: {len(spotify_csv_df):,} songs")
        
        # Check for columns not in primary
        if len(primary_df) > 0:
            new_cols = set(spotify_csv_df.columns) - set(primary_df.columns)
            if new_cols:
                print(f"    Additional columns in CSV: {new_cols}")
    else:
        print(f"  WARNING: Spotify CSV not found: {spotify_csv_path}")
        spotify_csv_df = pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # 3. Load secondary dataset (900K) - add unique songs
    # -------------------------------------------------------------------------
    secondary_path = DataPrepConfig.DATASET_DIR / DataPrepConfig.SECONDARY_DATASET
    if secondary_path.exists():
        secondary_df = load_jsonl(secondary_path, max_rows)
        secondary_df = normalize_columns(secondary_df)  # Standardize column names
        secondary_df = normalize_to_spotify_api(secondary_df)  # Convert 0-100 to 0-1
        print(f"  Secondary (900K): {len(secondary_df):,} songs (normalized)")
    else:
        print(f"  WARNING: Secondary dataset not found: {secondary_path}")
        secondary_df = pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # 4. Load tracks_features.csv - for audio feature enrichment
    # -------------------------------------------------------------------------
    tracks_path = DataPrepConfig.DATASET_DIR / DataPrepConfig.TRACKS_FEATURES
    if tracks_path.exists():
        tracks_df = load_csv(tracks_path, max_rows)
        print(f"  Tracks Features: {len(tracks_df):,} songs (audio features only)")
    else:
        print(f"  WARNING: Tracks features not found: {tracks_path}")
        tracks_df = pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # MERGE STRATEGY
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("MERGING DATASETS")
    print("-" * 60)
    
    # Start with primary dataset
    if len(primary_df) == 0:
        raise ValueError("Primary dataset is required but not found!")
    
    merged_df = primary_df.copy()
    
    # Create merge key (artist + song name, lowercase)
    # After normalize_columns: 'Artist(s)' -> 'artist', 'song' -> 'song_name'
    song_col = 'song_name' if 'song_name' in merged_df.columns else ('song' if 'song' in merged_df.columns else 'name')
    artist_col = 'artist' if 'artist' in merged_df.columns else ('Artist(s)' if 'Artist(s)' in merged_df.columns else 'artists')
    
    merged_df['_merge_key'] = (
        merged_df[artist_col].fillna('').str.lower().str.strip() + '|' +
        merged_df[song_col].fillna('').str.lower().str.strip()
    )
    existing_keys = set(merged_df['_merge_key'].unique())
    print(f"Starting with primary: {len(merged_df):,} songs")
    
    # Add unique songs from secondary (900K)
    if len(secondary_df) > 0:
        song_col_sec = 'song_name' if 'song_name' in secondary_df.columns else ('song' if 'song' in secondary_df.columns else 'name')
        artist_col_sec = 'artist' if 'artist' in secondary_df.columns else ('Artist(s)' if 'Artist(s)' in secondary_df.columns else 'artists')
        
        secondary_df['_merge_key'] = (
            secondary_df[artist_col_sec].fillna('').str.lower().str.strip() + '|' +
            secondary_df[song_col_sec].fillna('').str.lower().str.strip()
        )
        
        # Only add songs not already in primary
        unique_secondary = secondary_df[~secondary_df['_merge_key'].isin(existing_keys)]
        print(f"Adding {len(unique_secondary):,} unique songs from secondary (900K)")
        
        # Get common columns
        common_cols = list(set(merged_df.columns) & set(secondary_df.columns))
        merged_df = pd.concat([merged_df[common_cols], unique_secondary[common_cols]], ignore_index=True)
        existing_keys = set(merged_df['_merge_key'].unique())
    
    # -------------------------------------------------------------------------
    # ENRICH AUDIO FEATURES from tracks_features.csv
    # -------------------------------------------------------------------------
    if len(tracks_df) > 0:
        print("\n" + "-" * 60)
        print("ENRICHING AUDIO FEATURES FROM TRACKS_FEATURES.CSV")
        print("-" * 60)
        
        # Create merge key for tracks
        tracks_df['_merge_key'] = (
            tracks_df['artists'].fillna('').str.lower().str.strip() + '|' +
            tracks_df['name'].fillna('').str.lower().str.strip()
        )
        
        # Identify audio feature columns (already in 0-1 scale)
        track_audio_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        available_cols = [c for c in track_audio_cols if c in tracks_df.columns]
        print(f"  Found {len(available_cols)} audio features in tracks_features")
        
        # Create lookup dict from tracks_features (drop duplicates to avoid index error)
        tracks_unique = tracks_df.drop_duplicates(subset='_merge_key', keep='first')
        tracks_lookup = tracks_unique.set_index('_merge_key')[available_cols].to_dict('index')
        print(f"  Created lookup for {len(tracks_lookup):,} unique tracks (from {len(tracks_df):,})")
        
        # Fill missing audio features in merged_df
        filled_count = 0
        for feat in available_cols:
            if feat not in merged_df.columns:
                merged_df[feat] = np.nan
            
            # Find rows with missing values
            missing_mask = merged_df[feat].isna()
            missing_count = missing_mask.sum()
            
            if missing_count > 0:
                # Fill from tracks_lookup
                for idx in merged_df[missing_mask].index:
                    key = merged_df.loc[idx, '_merge_key']
                    if key in tracks_lookup and feat in tracks_lookup[key]:
                        merged_df.loc[idx, feat] = tracks_lookup[key][feat]
                        filled_count += 1
        
        new_missing = sum(merged_df[feat].isna().sum() for feat in available_cols)
        print(f"  Enriched {filled_count:,} missing audio feature values")
        print(f"  Remaining missing values: {new_missing:,}")
    
    # Remove merge key
    if '_merge_key' in merged_df.columns:
        merged_df = merged_df.drop('_merge_key', axis=1)
    
    print(f"\n{'=' * 60}")
    print(f"FINAL MERGED DATASET: {len(merged_df):,} songs")
    print(f"{'=' * 60}")
    
    # Free memory
    del secondary_df
    if 'tracks_df' in dir():
        del tracks_df
    gc.collect()
    
    return merged_df


def merge_datasets(primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy function - kept for compatibility.
    Use load_and_merge_all_datasets() instead.
    """
    return load_and_merge_all_datasets()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    print("\n" + "=" * 60)
    print("HANDLING MISSING VALUES")
    print("=" * 60)
    
    df = df.copy()
    initial_rows = len(df)
    
    # Get lyrics column
    lyrics_col = 'text' if 'text' in df.columns else 'lyrics'
    
    # Drop rows without lyrics (essential for BERT)
    if lyrics_col in df.columns:
        before = len(df)
        df = df[df[lyrics_col].notna() & (df[lyrics_col].str.len() > 50)]
        print(f"Dropped {before - len(df):,} rows without valid lyrics")
    
    # Handle missing audio features - fill with median
    for feat in DataPrepConfig.AUDIO_FEATURES:
        if feat in df.columns:
            null_count = df[feat].isnull().sum()
            if null_count > 0:
                median_val = df[feat].median()
                df[feat] = df[feat].fillna(median_val)
                print(f"  Filled {null_count:,} missing {feat} with median ({median_val:.3f})")
    
    # Handle missing emotion - fill with 'neutral'
    if 'emotion' in df.columns:
        null_count = df['emotion'].isnull().sum()
        if null_count > 0:
            df['emotion'] = df['emotion'].fillna('neutral')
            print(f"  Filled {null_count:,} missing emotion with 'neutral'")
    
    # Handle missing genre
    genre_col = 'Genre' if 'Genre' in df.columns else 'genre'
    if genre_col in df.columns:
        null_count = df[genre_col].isnull().sum()
        if null_count > 0:
            df[genre_col] = df[genre_col].fillna('Unknown')
            print(f"  Filled {null_count:,} missing genre with 'Unknown'")
    
    print(f"\nFinal: {len(df):,} rows (removed {initial_rows - len(df):,})")
    
    return df


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits."""
    print("\n" + "=" * 60)
    print("CREATING DATA SPLITS")
    print("=" * 60)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_seed
    )
    
    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        random_state=random_seed
    )
    
    print(f"Train: {len(train):,} ({len(train)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val):,} ({len(val)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test):,} ({len(test)/len(df)*100:.1f}%)")
    
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# =============================================================================
# 1.2 LYRICS PROCESSING & BERT EMBEDDINGS
# =============================================================================

import re

def clean_lyrics(text: str) -> str:
    """Clean and normalize lyrics text."""
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


class LyricsDataset(Dataset):
    """Dataset for batch processing lyrics through BERT."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'idx': idx
        }


def create_bert_embeddings(
    df: pd.DataFrame,
    output_path: Path,
    batch_size: int = 32,
    device: str = "cuda"
) -> np.ndarray:
    """
    Create BERT embeddings for all lyrics in the dataset.
    
    Uses [CLS] token embedding as sentence representation.
    """
    print("\n" + "=" * 60)
    print("CREATING BERT EMBEDDINGS")
    print("=" * 60)
    
    # Get lyrics column
    lyrics_col = 'text' if 'text' in df.columns else 'lyrics'
    
    # Clean lyrics
    print("Cleaning lyrics...")
    cleaned_lyrics = [clean_lyrics(text) for text in tqdm(df[lyrics_col].tolist())]
    
    # Load BERT
    print(f"Loading BERT model: {DataPrepConfig.BERT_MODEL}")
    tokenizer = BertTokenizer.from_pretrained(DataPrepConfig.BERT_MODEL)
    model = BertModel.from_pretrained(DataPrepConfig.BERT_MODEL)
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = LyricsDataset(cleaned_lyrics, tokenizer, DataPrepConfig.MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=False
    )
    
    # Generate embeddings
    print(f"Generating embeddings for {len(df):,} songs...")
    print(f"Device: {device}")
    
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Use [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved embeddings to: {output_path}")
    
    # Clear GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return embeddings


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_data_preparation(max_rows: Optional[int] = None):
    """
    Run the complete data preparation pipeline.
    
    Args:
        max_rows: Limit rows for testing (None for full dataset)
    """
    print("\n" + "=" * 60)
    print("HYBRID MUSIC RECOMMENDER - DATA PREPARATION PIPELINE")
    print("=" * 60)
    print(f"Device: {DataPrepConfig.DEVICE}")
    print(f"Max rows: {max_rows or 'ALL'}")
    
    # Create output directories
    DataPrepConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DataPrepConfig.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1.1 DATA LOADING & EXPLORATION
    # =========================================================================
    
    print("\n\n" + "#" * 60)
    print("# STEP 1.1: DATA LOADING & EXPLORATION")
    print("#" * 60)
    
    # Load and merge all 4 datasets
    merged_df = load_and_merge_all_datasets(max_rows)
    
    # Note: Normalization is done during load_and_merge_all_datasets()
    # - normalize_columns() standardizes column names
    # - normalize_to_spotify_api() converts 0-100 to 0-1 scale
    
    # Handle missing values
    cleaned_df = handle_missing_values(merged_df)
    
    # Free memory
    del merged_df
    gc.collect()
    
    # Create splits
    train_df, val_df, test_df = create_splits(
        cleaned_df,
        train_ratio=DataPrepConfig.TRAIN_RATIO,
        val_ratio=DataPrepConfig.VAL_RATIO,
        test_ratio=DataPrepConfig.TEST_RATIO,
        random_seed=DataPrepConfig.RANDOM_SEED
    )
    
    # Save splits
    print("\nSaving data splits...")
    train_df.to_csv(DataPrepConfig.OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(DataPrepConfig.OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(DataPrepConfig.OUTPUT_DIR / "test.csv", index=False)
    print(f"  Saved to: {DataPrepConfig.OUTPUT_DIR}")
    
    # Save dataset info
    dataset_info = {
        'total_songs': len(cleaned_df),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'columns': list(cleaned_df.columns),
        'audio_features': DataPrepConfig.AUDIO_FEATURES
    }
    with open(DataPrepConfig.OUTPUT_DIR / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # =========================================================================
    # 1.2 LYRICS PROCESSING & BERT EMBEDDINGS
    # =========================================================================
    
    print("\n\n" + "#" * 60)
    print("# STEP 1.2: LYRICS PROCESSING & BERT EMBEDDINGS")
    print("#" * 60)
    
    # Generate embeddings for each split
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\n--- Processing {split_name} split ---")
        
        embedding_path = DataPrepConfig.EMBEDDINGS_DIR / f"{split_name}_embeddings.npy"
        
        # Check if embeddings already exist
        if embedding_path.exists():
            print(f"Embeddings already exist: {embedding_path}")
            continue
        
        create_bert_embeddings(
            split_df,
            embedding_path,
            batch_size=DataPrepConfig.BATCH_SIZE,
            device=DataPrepConfig.DEVICE
        )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {DataPrepConfig.OUTPUT_DIR}")
    print(f"Files created:")
    for f in DataPrepConfig.OUTPUT_DIR.rglob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.relative_to(DataPrepConfig.OUTPUT_DIR)}: {size_mb:.1f} MB")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Preparation Pipeline")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit rows for testing (default: all)")
    parser.add_argument("--test", action="store_true",
                        help="Run with 1000 rows for quick test")
    
    args = parser.parse_args()
    
    max_rows = 1000 if args.test else args.max_rows
    
    run_data_preparation(max_rows=max_rows)
