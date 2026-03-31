"""
Audio Autoencoder (Model 2).

Architecture:
    ENCODER:  Input(9) -> FC(16) -> GELU -> FC(32)  [Bottleneck / Latent]
    DECODER:  Input(32) -> FC(16) -> GELU -> FC(9)   [Reconstruction]

Loss:  MSELoss(reconstruction, original)  -- fully Unsupervised.

The `encode()` method is the only part used during FAISS indexing and inference.
The Decoder is only used during training to guide representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 9, latent_dim: int = 32, dropout: float = 0.1):
        super().__init__()

        # ── Encoder  9 → 16 → 32 (latent) ────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(16, latent_dim),
        )

        # ── Decoder  32 → 16 → 9 ──────────────────────────────────────────────
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
        emb    = F.normalize(latent, p=2, dim=-1)  # L2-normalise for FAISS cosine
        return {'embedding': emb, 'reconstruction': recon}

    # ── Encode only (inference / FAISS building) ──────────────────────────────
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised latent vector (shape: [batch, latent_dim]).
        This is the only call used when building / searching the FAISS index.
        Signature is identical to the previous AudioOnlyModel.encode(),
        so CLIPAudioBridge needs no changes.
        """
        latent = self.encoder(x)
        return F.normalize(latent, p=2, dim=-1)
