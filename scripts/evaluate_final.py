"""
Final Evaluation on Test Set.

Generates final metrics:
- Accuracy (Emotion)
- Recommendation Quality (simulated)

Author: Graduation Project
Created: 2026-01-11
"""

import sys
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
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
        
        return {
            'embedding': embedding,
            'emotion_logits': self.emotion_head(embedding),
            'genre_logits': self.genre_head(embedding),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    config = get_config()
    data_dir = Path(config.paths.processed_data_dir)
    model_dir = Path(config.paths.model_dir)
    
    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv(data_dir / "test.csv")
    test_emb = np.load(data_dir / "embeddings" / "test_embeddings.npy")
    
    # Helper to load train stats for normalization
    train_data = pd.read_csv(data_dir / "train.csv")
    audio_cols = [c for c in config.audio.audio_features if c in train_data.columns]
    
    train_audio = train_data[audio_cols].values.astype(np.float32)
    audio_mean = train_audio.mean(axis=0)
    audio_std = train_audio.std(axis=0) + 1e-8
    
    # Process test audio
    test_audio = test_data[audio_cols].values.astype(np.float32)
    test_audio = (test_audio - audio_mean) / audio_std
    
    # Labels
    emotion_map = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'love': 4, 'surprise': 5}
    reverse_map = {v: k for k, v in emotion_map.items()}
    test_labels = [emotion_map.get(e.lower(), 0) for e in test_data['emotion'].fillna('joy').astype(str)]
    
    # Load model
    print("Loading model...")
    # Get num_genres needed for model init
    genre_col = 'genre' if 'genre' in train_data.columns else 'Genre'
    num_genres = train_data[genre_col].nunique()
    
    model = ImprovedModel(num_emotions=6, num_genres=num_genres, audio_dim=len(config.audio.audio_features))
    model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location=args.device))
    model.to(args.device)
    model.eval()
    
    # Prediction
    print("Running inference on Test Set...")
    all_preds_emo = []
    
    batch_size = 512
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size)):
            bert = torch.tensor(test_emb[i:i+batch_size], dtype=torch.float32).to(args.device)
            audio = torch.tensor(test_audio[i:i+batch_size], dtype=torch.float32).to(args.device)
            
            out = model(bert, audio)
            preds = out['emotion_logits'].argmax(1).cpu().numpy()
            all_preds_emo.extend(preds)
    
    # Metrics
    acc = accuracy_score(test_labels, all_preds_emo)
    print("\n" + "=" * 60)
    print(f"TEST ACCURACY (EMOTION): {acc:.4f} ({acc*100:.2f}%)")
    print("=" * 60)
    
    print("\nClassification Report:")
    print(classification_report(test_labels, all_preds_emo, target_names=[reverse_map[i] for i in range(6)]))
    
    # Save results
    with open(model_dir / "evaluation_results.txt", "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(classification_report(test_labels, all_preds_emo, target_names=[reverse_map[i] for i in range(6)]))
    
    print(f"Results saved to {model_dir / 'evaluation_results.txt'}")


if __name__ == "__main__":
    main()
