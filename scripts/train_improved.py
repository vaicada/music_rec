"""
Improved Training Script - Higher Accuracy Version.

Improvements from train_final.py:
1. More epochs (50 instead of 20)
2. Label smoothing (reduces overfitting)
3. Class weighting (handles imbalanced data)
4. Learning rate warmup
5. Better dropout scheduling

Target: 65-70% emotion accuracy

Author: Graduation Project
Created: 2026-01-11
"""

import os
import sys
import gc
import math
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_music_engine import get_config


class MusicDataset(Dataset):
    """Dataset with class weights computation."""
    
    def __init__(self, csv_path, embeddings_path, config, fit_from=None):
        print(f"Loading {csv_path}...")
        self.data = pd.read_csv(csv_path)
        self.embeddings = np.load(embeddings_path)
        
        # Audio features
        audio_cols = [c for c in config.audio.audio_features if c in self.data.columns]
        audio_vals = self.data[audio_cols].values.astype(np.float32)
        
        if fit_from is None:
            self.audio_mean = audio_vals.mean(axis=0)
            self.audio_std = audio_vals.std(axis=0) + 1e-8
        else:
            self.audio_mean = fit_from.audio_mean
            self.audio_std = fit_from.audio_std
        
        self.audio_features = (audio_vals - self.audio_mean) / self.audio_std
        
        # Emotion labels
        self.emotion_map = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'love': 4, 'surprise': 5}
        emotions = self.data['emotion'].fillna('joy').astype(str)
        self.emotion_labels = [self.emotion_map.get(e.lower(), 0) for e in emotions]
        self.num_emotions = len(self.emotion_map)
        
        # Genre labels
        genre_col = 'genre' if 'genre' in self.data.columns else 'Genre'
        genres = self.data[genre_col].fillna('Unknown').astype(str).tolist()
        
        if fit_from is None:
            unique_genres = sorted(set(genres))
            self.genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
        else:
            self.genre_to_idx = fit_from.genre_to_idx
        
        self.genre_labels = [self.genre_to_idx.get(g, 0) for g in genres]
        self.num_genres = len(self.genre_to_idx)
        
        # Compute class weights for emotion (inverse frequency)
        if fit_from is None:
            emotion_counts = Counter(self.emotion_labels)
            total = sum(emotion_counts.values())
            self.emotion_weights = torch.tensor([
                total / (self.num_emotions * emotion_counts.get(i, 1))
                for i in range(self.num_emotions)
            ], dtype=torch.float32)
            # Normalize weights
            self.emotion_weights = self.emotion_weights / self.emotion_weights.sum() * self.num_emotions
            
            print(f"Emotion class weights: {self.emotion_weights.tolist()}")
        else:
            self.emotion_weights = fit_from.emotion_weights
        
        print(f"  {len(self.data)} samples, {self.num_emotions} emotions, {self.num_genres} genres")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'embedding': torch.tensor(self.embeddings[idx], dtype=torch.float32),
            'audio': torch.tensor(self.audio_features[idx], dtype=torch.float32),
            'emotion': self.emotion_labels[idx],
            'genre': self.genre_labels[idx],
        }


class ImprovedModel(nn.Module):
    """Improved model with better regularization."""
    
    def __init__(self, num_emotions, num_genres, audio_dim=9, dropout=0.3):
        super().__init__()
        
        # BERT projection with more capacity
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
        
        # Audio encoder with residual-like connections
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
        
        # Fusion with more layers
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
        
        # Classification heads with hidden layer
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
        
        params = sum(p.numel() for p in self.parameters())
        print(f"Model: {params:,} parameters")
    
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


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing (no class weighting)."""
    
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        
        # One-hot encode target
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        
        # Apply label smoothing
        smoothed = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        # Compute loss
        log_prob = F.log_softmax(pred, dim=-1)
        loss = -(smoothed * log_prob).sum(dim=-1)
        
        return loss.mean()


def get_lr_with_warmup(optimizer, epoch, warmup_epochs, initial_lr, max_lr):
    """Learning rate with warmup."""
    if epoch < warmup_epochs:
        # Linear warmup
        lr = initial_lr + (max_lr - initial_lr) * epoch / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (50 - warmup_epochs)
        lr = initial_lr + 0.5 * (max_lr - initial_lr) * (1 + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def train_epoch(model, loader, optimizer, emotion_loss_fn, genre_loss_fn, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    emotion_correct = 0
    genre_correct = 0
    total = 0
    
    # Adjust genre weight based on epoch (start low, increase over time)
    genre_weight = min(0.4, 0.2 + 0.2 * epoch / total_epochs)
    emotion_weight = 1 - genre_weight
    
    pbar = tqdm(loader, desc=f"Train Ep{epoch+1}")
    for batch in pbar:
        emb = batch['embedding'].to(device)
        audio = batch['audio'].to(device)
        emotion = batch['emotion'].to(device)
        genre = batch['genre'].to(device)
        
        optimizer.zero_grad()
        
        out = model(emb, audio)
        
        loss_e = emotion_loss_fn(out['emotion_logits'], emotion)
        loss_g = genre_loss_fn(out['genre_logits'], genre)
        loss = emotion_weight * loss_e + genre_weight * loss_g
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        emotion_correct += (out['emotion_logits'].argmax(1) == emotion).sum().item()
        genre_correct += (out['genre_logits'].argmax(1) == genre).sum().item()
        total += len(emotion)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'emo': f'{emotion_correct/total:.2%}'
        })
    
    return {
        'loss': total_loss / len(loader),
        'emotion_acc': emotion_correct / total,
        'genre_acc': genre_correct / total,
    }


def validate(model, loader, emotion_loss_fn, device):
    model.eval()
    total_loss = 0
    emotion_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            emb = batch['embedding'].to(device)
            audio = batch['audio'].to(device)
            emotion = batch['emotion'].to(device)
            
            out = model(emb, audio)
            loss = emotion_loss_fn(out['emotion_logits'], emotion)
            
            total_loss += loss.item()
            emotion_correct += (out['emotion_logits'].argmax(1) == emotion).sum().item()
            total += len(emotion)
    
    return {
        'val_loss': total_loss / len(loader),
        'val_emotion_acc': emotion_correct / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("=" * 60)
    print("IMPROVED TRAINING - Higher Accuracy Version")
    print("=" * 60)
    print(f"Time: {datetime.now()}")
    print(f"Epochs: {args.epochs}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    config = get_config()
    data_dir = Path(config.paths.processed_data_dir)
    model_dir = Path(config.paths.model_dir)
    model_dir.mkdir(exist_ok=True)
    
    # Create datasets
    print("\n[1/3] Loading datasets...")
    train_ds = MusicDataset(
        data_dir / "train.csv",
        data_dir / "embeddings" / "train_embeddings.npy",
        config
    )
    val_ds = MusicDataset(
        data_dir / "val.csv",
        data_dir / "embeddings" / "val_embeddings.npy",
        config,
        fit_from=train_ds
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Create model
    print("\n[2/3] Creating model...")
    model = ImprovedModel(
        num_emotions=train_ds.num_emotions,
        num_genres=train_ds.num_genres,
        audio_dim=len(config.audio.audio_features),
        dropout=args.dropout
    )
    model.to(args.device)
    
    # Loss functions - simple label smoothing, no class weighting
    emotion_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    genre_loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Train
    print("\n[3/3] Training...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    best_val_acc = 0
    history = []
    patience = 10
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Update learning rate with warmup
        lr = get_lr_with_warmup(optimizer, epoch, args.warmup_epochs, 1e-5, args.lr)
        
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={lr:.6f})")
        
        train_m = train_epoch(model, train_loader, optimizer, 
                              emotion_loss_fn, genre_loss_fn, args.device, epoch, args.epochs)
        val_m = validate(model, val_loader, emotion_loss_fn, args.device)
        
        print(f"Train: loss={train_m['loss']:.4f}, emo_acc={train_m['emotion_acc']:.2%}, genre_acc={train_m['genre_acc']:.2%}")
        print(f"Val: loss={val_m['val_loss']:.4f}, emo_acc={val_m['val_emotion_acc']:.2%}")
        
        history.append({**train_m, **val_m, 'epoch': epoch+1, 'lr': lr})
        
        # Save best model based on accuracy (not loss)
        if val_m['val_emotion_acc'] > best_val_acc:
            best_val_acc = val_m['val_emotion_acc']
            best_val_loss = val_m['val_loss']
            torch.save(model.state_dict(), model_dir / "best_model.pth")
            print(f">> New best model! Accuracy: {best_val_acc:.2%}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # Save final
    torch.save(model.state_dict(), model_dir / "final_model.pth")
    
    import json
    with open(model_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Best val accuracy: {best_val_acc:.2%}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {model_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
