"""
Audio Autoencoder (Model 2).

Architecture:
    ENCODER:  Input(9) -> FC(16) -> GELU -> FC(8)   [Bottleneck / Latent]
    DECODER:  Input(8)  -> FC(16) -> GELU -> FC(9)   [Reconstruction]

Loss:  MSELoss(reconstruction, original)  -- fully Unsupervised.

The `encode()` method is the only part used during FAISS indexing and inference.
The Decoder is only used during training to guide representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 9, latent_dim: int = 8, dropout: float = 0.1):
        super().__init__()

        # ── Encoder  9 → 16 → 8 (latent) ────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(16, latent_dim),
        )

        # ── Decoder  8 → 16 → 9 ──────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(16, input_dim),
        )

    # ── Forward (training) ────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """Returns dict with latent embedding and reconstruction."""
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        return {'embedding': latent, 'reconstruction': recon}

    # ── Encode only (inference / FAISS building) ──────────────────────────────
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw latent vector (shape: [batch, latent_dim]).
        Uses L2 distance (Euclidean) in FAISS instead of cosine similarity.
        Raw embeddings preserve natural spacing between songs.
        """
        return self.encoder(x)

