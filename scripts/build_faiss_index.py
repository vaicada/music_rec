"""
Build FAISS Index for Similarity Search.

This script:
1. Loads the trained model
2. Generates embeddings for all songs
3. Builds FAISS index
4. Saves index for inference

Author: Graduation Project
Created: 2026-01-11
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_music_engine import get_config


class ImprovedModel(nn.Module):
    """Improved model architecture matching train_improved.py."""
    
    def __init__(self, num_genres, num_emotions=6, audio_dim=9, dropout=0.3):
        super().__init__()
        
        self.bert_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(384, 256),
            nn.LayerNorm(256)
        )
        
        self.audio_enc = nn.Sequential(
            nn.Linear(audio_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.LayerNorm(64)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(320, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_emotions)
        )
        
        self.genre_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_genres)
        )
    
    def forward(self, bert_emb, audio):
        b = self.bert_proj(bert_emb)
        a = self.audio_enc(audio)
        
        combined = torch.cat([b, a], dim=-1)
        embedding = self.fusion(combined)
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        # Return just embedding for FAISS building
        return embedding


def load_data(csv_path, embeddings_path, config, audio_mean=None, audio_std=None):
    """Load and preprocess data."""
    print(f"Loading {csv_path}...")
    data = pd.read_csv(csv_path)
    embeddings = np.load(embeddings_path)
    
    # Audio features
    audio_cols = [c for c in config.audio.audio_features if c in data.columns]
    audio_vals = data[audio_cols].values.astype(np.float32)
    
    if audio_mean is None:
        audio_mean = audio_vals.mean(axis=0)
        audio_std = audio_vals.std(axis=0) + 1e-8
    
    audio_features = (audio_vals - audio_mean) / audio_std
    
    return data, embeddings, audio_features, audio_mean, audio_std


def generate_embeddings(model, bert_embeddings, audio_features, device, batch_size=512):
    """Generate 64D embeddings for all samples."""
    model.eval()
    all_embeddings = []
    
    n_samples = len(bert_embeddings)
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Generating embeddings"):
            batch_bert = torch.tensor(
                bert_embeddings[i:i+batch_size], 
                dtype=torch.float32
            ).to(device)
            
            batch_audio = torch.tensor(
                audio_features[i:i+batch_size],
                dtype=torch.float32
            ).to(device)
            
            embeddings = model(batch_bert, batch_audio)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def build_faiss_index(embeddings, use_gpu=True, index_type='L2'):
    """Build FAISS index from embeddings."""
    print(f"Building FAISS index ({index_type})...")
    dim = embeddings.shape[1]
    
    if index_type == 'L2':
        index = faiss.IndexFlatL2(dim)
    elif index_type == 'IP':  # Inner Product (cosine similarity for normalized vectors)
        index = faiss.IndexFlatIP(dim)
    elif index_type == 'HNSW':
        index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
    else:
        index = faiss.IndexFlatL2(dim)
    
    # Try to use GPU
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Moving index to GPU...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Add embeddings
    print(f"Adding {len(embeddings)} vectors to index...")
    index.add(embeddings.astype(np.float32))
    
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--index-type', type=str, default='IP', choices=['L2', 'IP', 'HNSW'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    
    print("=" * 60)
    print("BUILDING FAISS INDEX")
    print("=" * 60)
    print(f"Time: {datetime.now()}")
    print(f"Device: {args.device}")
    print(f"Index type: {args.index_type}")
    
    config = get_config()
    data_dir = Path(config.paths.processed_data_dir)
    model_dir = Path(config.paths.model_dir)
    
    # Load model
    model_path = args.model_path or (model_dir / "best_model.pth")
    print(f"\n[1/4] Loading model from {model_path}...")
    
    # First, load training data to get num_genres
    train_data = pd.read_csv(data_dir / "train.csv")
    genre_col = 'genre' if 'genre' in train_data.columns else 'Genre'
    num_genres = train_data[genre_col].nunique()
    
    model = ImprovedModel(num_genres=num_genres, audio_dim=len(config.audio.audio_features))
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    print(f"Model loaded with {num_genres} genres")
    
    # Load all data (train + val + test)
    print("\n[2/4] Loading all data...")
    
    # Train
    train_data, train_bert, train_audio, audio_mean, audio_std = load_data(
        data_dir / "train.csv",
        data_dir / "embeddings" / "train_embeddings.npy",
        config
    )
    
    # Val
    val_data, val_bert, val_audio, _, _ = load_data(
        data_dir / "val.csv",
        data_dir / "embeddings" / "val_embeddings.npy",
        config,
        audio_mean, audio_std
    )
    
    # Test
    test_data, test_bert, test_audio, _, _ = load_data(
        data_dir / "test.csv",
        data_dir / "embeddings" / "test_embeddings.npy",
        config,
        audio_mean, audio_std
    )
    
    # Combine all
    all_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
    all_bert = np.vstack([train_bert, val_bert, test_bert])
    all_audio = np.vstack([train_audio, val_audio, test_audio])
    
    print(f"Total songs: {len(all_data):,}")
    
    # Generate embeddings
    print("\n[3/4] Generating 64D embeddings...")
    all_embeddings = generate_embeddings(
        model, all_bert, all_audio, args.device, args.batch_size
    )
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    # Build index
    print("\n[4/4] Building FAISS index...")
    index = build_faiss_index(all_embeddings, use_gpu=False, index_type=args.index_type)
    
    # Save
    # Convert to CPU index for saving if on GPU
    if hasattr(index, 'index'):
        index = faiss.index_gpu_to_cpu(index)
    
    index_path = model_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    print(f"Index saved to: {index_path}")
    
    # Save mappings for inference engine (CRITICAL!)
    import pickle
    song_ids = []
    idx_to_song = {}
    song_col = 'song_name' if 'song_name' in all_data.columns else 'song'
    artist_col = 'artist' if 'artist' in all_data.columns else 'Artist(s)'
    
    for idx, row in all_data.iterrows():
        song_info = {
            'idx': idx,
            'song': row.get(song_col, f'Unknown_{idx}'),
            'artist': row.get(artist_col, 'Unknown'),
            'genre': row.get('genre', row.get('Genre', '')),
            'emotion': row.get('emotion', ''),
            'album': row.get('album', row.get('Album', '')),
        }
        song_id = f"{song_info['artist']}|{song_info['song']}"
        song_ids.append(song_id)
        idx_to_song[idx] = song_info
    
    mappings_path = str(index_path) + '.mappings.pkl'
    with open(mappings_path, 'wb') as f:
        pickle.dump({'song_ids': song_ids, 'idx_to_song': idx_to_song}, f)
    print(f"Mappings saved to: {mappings_path}")
    
    # Save embeddings
    embeddings_path = model_dir / "all_embeddings.npy"
    np.save(embeddings_path, all_embeddings)
    print(f"Embeddings saved to: {embeddings_path}")
    
    # Save song metadata for lookup
    metadata_path = model_dir / "song_metadata.csv"
    all_data[['song_name', 'artist', 'genre', 'emotion']].to_csv(metadata_path, index=False)
    print(f"Metadata saved to: {metadata_path}")
    
    # Test search
    print("\n" + "=" * 60)
    print("Testing search...")
    
    # Search with first song
    query_idx = 0
    query_embedding = all_embeddings[query_idx:query_idx+1]
    
    distances, indices = index.search(query_embedding, 6)  # Top 5 + self
    
    query_song = all_data.iloc[query_idx]
    print(f"\nQuery: {query_song['song_name']} by {query_song['artist']}")
    print(f"Genre: {query_song['genre']}, Emotion: {query_song['emotion']}")
    print("\nTop 5 similar songs:")
    
    for i, (dist, idx) in enumerate(zip(distances[0][1:], indices[0][1:])):
        similar = all_data.iloc[idx]
        print(f"  {i+1}. {similar['song_name']} by {similar['artist']}")
        print(f"     Genre: {similar['genre']}, Emotion: {similar['emotion']}")
        print(f"     Similarity: {1/(1+dist):.4f}" if args.index_type == 'L2' else f"     Similarity: {dist:.4f}")
    
    print("\n" + "=" * 60)
    print("FAISS INDEX BUILD COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
