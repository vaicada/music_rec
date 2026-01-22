"""
Training Module for the Hybrid Music Recommendation Engine.

This module contains the training loop, triplet mining, and model optimization
for the hybrid multi-modal model.

Author: Graduation Project
Created: 2026-01-06
"""

import os
import time
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from .logger import get_logger, log_step
from .config import Config, get_config
from .model import HybridMusicModel, MultiTaskLoss, TripletLoss
from .processors import DataManager, MusicDataset


class TripletMiner:
    """
    Mines triplets (anchor, positive, negative) for training.
    
    Uses the similar songs in the dataset as positives.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[Config] = None
    ):
        """
        Initialize the TripletMiner.
        
        Args:
            data: DataFrame with similar song information.
            config: Configuration object.
        """
        self.config = config or get_config()
        self.logger = get_logger(self.config.paths.log_file)
        self.data = data.reset_index(drop=True)
        
        # Build song-to-index mapping
        self.song_to_idx: Dict[str, int] = {}
        song_col = 'song' if 'song' in data.columns else 'name'
        artist_col = 'Artist(s)' if 'Artist(s)' in data.columns else 'artists'
        
        for idx, row in data.iterrows():
            key = self._make_key(row.get(artist_col, ''), row.get(song_col, ''))
            self.song_to_idx[key] = idx
        
        # Build genre-to-songs mapping for negative sampling
        self.genre_to_indices: Dict[str, List[int]] = {}
        genre_col = 'Genre' if 'Genre' in data.columns else 'genre'
        
        if genre_col in data.columns:
            for idx, genre in enumerate(data[genre_col]):
                if pd.notna(genre):
                    genre_str = str(genre).lower()
                    if genre_str not in self.genre_to_indices:
                        self.genre_to_indices[genre_str] = []
                    self.genre_to_indices[genre_str].append(idx)
        
        self.all_indices = list(range(len(data)))
        
        self.logger.log(
            "TripletMiner initialized",
            "DATA",
            details={
                "total_songs": len(data),
                "genres": len(self.genre_to_indices)
            }
        )
    
    def _make_key(self, artist: str, song: str) -> str:
        """Create a unique key for a song."""
        return f"{str(artist).lower().strip()}|{str(song).lower().strip()}"
    
    def get_positive(self, anchor_idx: int) -> Optional[int]:
        """
        Get a positive sample index for an anchor.
        
        Args:
            anchor_idx: Index of the anchor song.
        
        Returns:
            Index of a similar song, or None if not found.
        """
        row = self.data.iloc[anchor_idx]
        
        # Try to find similar songs from the dataset
        similar_cols = [
            ('Similar Artist 1', 'Similar Song 1'),
            ('Similar Artist 2', 'Similar Song 2'),
            ('Similar Artist 3', 'Similar Song 3'),
        ]
        
        for artist_col, song_col in similar_cols:
            if artist_col in row and song_col in row:
                artist = row.get(artist_col)
                song = row.get(song_col)
                if pd.notna(artist) and pd.notna(song):
                    key = self._make_key(artist, song)
                    if key in self.song_to_idx:
                        return self.song_to_idx[key]
        
        # Fallback: same genre
        genre_col = 'Genre' if 'Genre' in self.data.columns else 'genre'
        if genre_col in row:
            genre = str(row[genre_col]).lower()
            if genre in self.genre_to_indices:
                candidates = [
                    idx for idx in self.genre_to_indices[genre] 
                    if idx != anchor_idx
                ]
                if candidates:
                    return random.choice(candidates)
        
        return None
    
    def get_negative(self, anchor_idx: int) -> int:
        """
        Get a negative sample index for an anchor.
        
        Args:
            anchor_idx: Index of the anchor song.
        
        Returns:
            Index of a dissimilar song.
        """
        row = self.data.iloc[anchor_idx]
        genre_col = 'Genre' if 'Genre' in self.data.columns else 'genre'
        
        if genre_col in row:
            anchor_genre = str(row[genre_col]).lower()
            
            # Get songs from different genres
            other_genres = [
                g for g in self.genre_to_indices.keys() 
                if g != anchor_genre
            ]
            
            if other_genres:
                chosen_genre = random.choice(other_genres)
                return random.choice(self.genre_to_indices[chosen_genre])
        
        # Fallback: random song
        candidates = [idx for idx in self.all_indices if idx != anchor_idx]
        return random.choice(candidates)
    
    def mine_triplets(self, batch_indices: List[int]) -> List[Tuple[int, int, int]]:
        """
        Mine triplets for a batch.
        
        Args:
            batch_indices: List of anchor indices.
        
        Returns:
            List of (anchor, positive, negative) index tuples.
        """
        triplets = []
        
        for anchor_idx in batch_indices:
            positive_idx = self.get_positive(anchor_idx)
            if positive_idx is None:
                continue
            
            negative_idx = self.get_negative(anchor_idx)
            triplets.append((anchor_idx, positive_idx, negative_idx))
        
        return triplets


class Trainer:
    """
    Training orchestrator for the Hybrid Music Model.
    
    Handles the complete training pipeline including:
    - Triplet mining
    - Multi-task learning
    - Learning rate scheduling
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        model: HybridMusicModel,
        data_manager: DataManager,
        config: Optional[Config] = None
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: HybridMusicModel instance.
            data_manager: DataManager instance.
            config: Configuration object.
        """
        self.config = config or get_config()
        self.logger = get_logger(self.config.paths.log_file)
        
        self.model = model
        self.data_manager = data_manager
        
        # Determine device (auto-detect if "auto")
        device_str = self.config.training.get_device()
        self.device = torch.device(device_str)
        self.model.to(self.device)
        
        # Log GPU info if using CUDA
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.logger.log(
                f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)",
                "TRAINING",
                level="SUCCESS"
            )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
        # Loss function
        self.loss_fn = MultiTaskLoss(config)
        self.triplet_loss_fn = TripletLoss(margin=self.config.training.triplet_margin)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics history
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
        
        self.logger.log_section("Trainer Initialized")
        self.logger.log(
            "Trainer ready",
            "TRAINING",
            details={
                "device": str(self.device),
                "optimizer": self.config.training.optimizer,
                "learning_rate": self.config.training.learning_rate
            }
        )
    
    def _create_optimizer(self) -> AdamW:
        """Create optimizer with parameter groups."""
        # Different learning rates for BERT vs other components
        bert_params = list(self.model.bert_encoder.bert.parameters())
        other_params = [
            p for n, p in self.model.named_parameters()
            if 'bert_encoder.bert' not in n
        ]
        
        param_groups = [
            {'params': bert_params, 'lr': self.config.training.learning_rate * 0.1},
            {'params': other_params, 'lr': self.config.training.learning_rate}
        ]
        
        return AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay,
            eps=self.config.training.adam_epsilon
        )
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    @log_step("TRAINING", "Training Epoch")
    def train_epoch(
        self,
        train_loader: DataLoader,
        triplet_miner: TripletMiner
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            triplet_miner: TripletMiner instance.
        
        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        total_triplet_loss = 0.0
        total_emotion_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids'),
                audio_features=batch['audio_features'],
                genre_idx=batch['genre_idx'],
                key_idx=batch['key_idx'],
                emotion_idx=batch['emotion_idx']
            )
            
            embeddings = outputs['embedding']
            emotion_logits = outputs['emotion_logits']
            
            # Compute losses
            # Simple contrastive loss within batch
            anchor_emb = embeddings
            positive_emb = embeddings.roll(1, 0)  # Simplified positive sampling
            negative_emb = embeddings.roll(-1, 0)  # Simplified negative sampling
            
            triplet_loss = self.triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
            emotion_loss = nn.CrossEntropyLoss()(emotion_logits, batch['emotion_label'])
            
            # Combined loss
            loss = (
                self.config.training.triplet_loss_weight * triplet_loss +
                self.config.training.emotion_loss_weight * emotion_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip_value
            )
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_triplet_loss += triplet_loss.item()
            total_emotion_loss += emotion_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'triplet': f'{triplet_loss.item():.4f}',
                'emotion': f'{emotion_loss.item():.4f}'
            })
        
        metrics = {
            'loss': total_loss / num_batches,
            'triplet_loss': total_triplet_loss / num_batches,
            'emotion_loss': total_emotion_loss / num_batches
        }
        
        # Log metrics
        for name, value in metrics.items():
            self.logger.log_metric(f"train_{name}", value, self.current_epoch + 1)
        
        return metrics
    
    @log_step("TRAINING", "Validation")
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader.
        
        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_triplet_loss = 0.0
        total_emotion_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids'),
                    audio_features=batch['audio_features'],
                    genre_idx=batch['genre_idx'],
                    key_idx=batch['key_idx'],
                    emotion_idx=batch['emotion_idx']
                )
                
                embeddings = outputs['embedding']
                emotion_logits = outputs['emotion_logits']
                
                # Compute losses
                anchor_emb = embeddings
                positive_emb = embeddings.roll(1, 0)
                negative_emb = embeddings.roll(-1, 0)
                
                triplet_loss = self.triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
                emotion_loss = nn.CrossEntropyLoss()(emotion_logits, batch['emotion_label'])
                
                loss = (
                    self.config.training.triplet_loss_weight * triplet_loss +
                    self.config.training.emotion_loss_weight * emotion_loss
                )
                
                # Accuracy
                predictions = emotion_logits.argmax(dim=-1)
                correct_predictions += (predictions == batch['emotion_label']).sum().item()
                total_samples += len(batch['emotion_label'])
                
                total_loss += loss.item()
                total_triplet_loss += triplet_loss.item()
                total_emotion_loss += emotion_loss.item()
                num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'triplet_loss': total_triplet_loss / num_batches,
            'emotion_loss': total_emotion_loss / num_batches,
            'emotion_accuracy': correct_predictions / total_samples
        }
        
        # Log metrics
        for name, value in metrics.items():
            self.logger.log_metric(f"val_{name}", value, self.current_epoch + 1)
        
        return metrics
    
    @log_step("TRAINING", "Full Training Run")
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        epochs: Optional[int] = None
    ) -> Dict[str, List]:
        """
        Run the complete training pipeline.
        
        Args:
            train_data: Training DataFrame.
            val_data: Validation DataFrame.
            epochs: Number of epochs (uses config default if None).
        
        Returns:
            Training history dictionary.
        """
        epochs = epochs or self.config.training.epochs
        
        self.logger.log_section("Training Started")
        self.logger.log_config(self.config.to_dict())
        
        # Create data loaders
        train_loader = self.data_manager.create_dataloader(train_data, shuffle=True)
        val_loader = self.data_manager.create_dataloader(val_data, shuffle=False)
        
        # Create triplet miner
        triplet_miner = TripletMiner(train_data, self.config)
        
        # Create scheduler
        num_training_steps = len(train_loader) * epochs
        self._create_scheduler(num_training_steps)
        
        self.logger.log(
            "Training configuration",
            "TRAINING",
            details={
                "epochs": epochs,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "batch_size": self.config.training.batch_size,
                "total_steps": num_training_steps
            }
        )
        
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            self.logger.log(f"Starting Epoch {epoch + 1}/{epochs}", "TRAINING")
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, triplet_miner)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.val_history.append(val_metrics)
            
            epoch_time = time.time() - start_time
            
            # Log epoch summary
            self.logger.log(
                f"Epoch {epoch + 1} completed",
                "TRAINING",
                details={
                    "train_loss": f"{train_metrics['loss']:.4f}",
                    "val_loss": f"{val_metrics['loss']:.4f}",
                    "val_accuracy": f"{val_metrics['emotion_accuracy']:.4f}",
                    "time_seconds": f"{epoch_time:.1f}"
                }
            )
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                # Save best model
                if self.config.training.save_best_only:
                    model_path = os.path.join(
                        self.config.paths.model_dir,
                        "best_model.pth"
                    )
                    os.makedirs(self.config.paths.model_dir, exist_ok=True)
                    self.model.save(model_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    self.logger.log(
                        f"Early stopping at epoch {epoch + 1}",
                        "TRAINING",
                        level="WARNING"
                    )
                    break
        
        self.logger.log_section("Training Completed")
        self.logger.log(
            "Training finished",
            "TRAINING",
            details={
                "total_epochs": self.current_epoch + 1,
                "best_val_loss": f"{self.best_val_loss:.4f}"
            },
            level="SUCCESS"
        )
        
        return {
            'train': self.train_history,
            'val': self.val_history
        }
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config.to_dict()
        }
        torch.save(checkpoint, path)
        self.logger.log(f"Checkpoint saved to {path}", "TRAINING")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        self.logger.log(f"Checkpoint loaded from {path}", "TRAINING")


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print("Use Trainer class to train the HybridMusicModel.")
