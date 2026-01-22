"""
Neural Network Architecture for the Hybrid Music Recommendation Engine.

This module implements the improved hybrid architecture used in training,
adapted for end-to-end inference.

Architecture Matches 'ImprovedModel' from training:
- BERT Branch: BertModel (frozen/infer) -> 768 -> 512 -> 384 -> 256
- Audio Branch: 9 -> 64 -> 128 -> 64
- Fusion: Concat -> 320 -> 256 -> 192 -> 128 -> 64

Author: Graduation Project
Created: 2026-01-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import BertModel

from .logger import get_logger
from .config import Config, get_config


class HybridMusicModel(nn.Module):
    """
    Improved Hybrid Music Model for Inference.
    
    Includes the raw BERT model to generate embeddings from text on-the-fly,
    then feeds them into the trained projection layers.
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__()
        self.config = config or get_config()
        self.logger = get_logger(self.config.paths.log_file)
        
        # Hyperparams matching training
        dropout = 0.3
        num_emotions = 6
        # Default num_genres, will be resized if loading checkpoint forces it
        num_genres = 3010 
        
        # 1. BERT Encoder (Feature Extractor)
        # Not present in trained checkpoint, but needed for inference
        self.bert = BertModel.from_pretrained(
            self.config.bert.model_name,
            cache_dir=self.config.paths.bert_cache_dir
        )
        
        # 2. Main Architecture (Matches trained weights)
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
            nn.Linear(self.config.audio.input_dim, 64),
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
        
        self.logger.log("HybridMusicModel (Inference Ready) initialized", "MODEL")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_features: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Run BERT to get embeddings
        with torch.no_grad():
             bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            )
             # Use CLS token
             bert_emb = bert_outputs.last_hidden_state[:, 0, :]
        
        # 2. Run Trained Layers
        b = self.bert_proj(bert_emb)
        a = self.audio_enc(audio_features)
        
        combined = torch.cat([b, a], dim=-1)
        embedding = self.fusion(combined)
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return {
            'embedding': embedding
        }

    def get_embedding(self, **kwargs) -> torch.Tensor:
        """Alias for forward to match inference API."""
        return self.forward(**kwargs)['embedding']

    def load(self, path: str, device: str = "cpu") -> None:
        """Robust loading function."""
        state_dict = torch.load(path, map_location=device)
        
        # Handle genre head size mismatch if needed
        if 'genre_head.3.weight' in state_dict:
             checkpoint_genres = state_dict['genre_head.3.weight'].shape[0]
             if checkpoint_genres != self.genre_head[3].out_features:
                 print(f"Resizing genre head from {self.genre_head[3].out_features} to {checkpoint_genres}")
                 self.genre_head[3] = nn.Linear(128, checkpoint_genres)
        
        # Load with strict=False to ignore missing 'bert' keys
        keys = self.load_state_dict(state_dict, strict=False)
        
        # Verify that the missing keys are ONLY the BERT ones (which is expected)
        missing_critical = [k for k in keys.missing_keys if not k.startswith('bert.')]
        if missing_critical:
            print(f"WARNING: Missing critical keys: {missing_critical}")
        
        self.logger.log(f"Model loaded from {path}", "MODEL", level="SUCCESS")
