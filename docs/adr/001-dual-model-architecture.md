# ADR-001: Dual-Model Architecture

## Status
Accepted

## Date
2026-01-06 (initial), updated 2026-03-16 (Model 2 added)

## Context

The Music Recommendation System needs to handle two distinct use cases:

1. **Rich recommendation** — when song lyrics, metadata (genre, emotion), and audio
   features are all available. This covers the ~551K curated songs in our training
   dataset that have been scraped with Genius lyrics.

2. **Audio-only recommendation** — when we only have Spotify audio features
   (energy, danceability, valence, tempo, etc.) from the `tracks_features.csv`
   dataset (~105K tracks). These tracks may lack lyrics or metadata.

A single model cannot serve both cases optimally:
- A BERT-based model requires text input; it produces poor embeddings when
  lyrics are empty.
- Purely audio-based models ignore the semantic richness of lyrics and
  emotion tags.

## Decision

We adopt a **dual-model architecture**:

### Model 1: Hybrid Multi-Modal (BERT + Audio + Metadata)

```
Input: Lyrics → BERT-base (768D) → BertEncoder (768→256)
       Audio → 11 features → AudioBranch (11→128→64)
       Metadata → Genre/Key/Emotion embeddings → MetadataBranch (→64)
       
Fusion: concat(256 + 64 + 64) = 384 → FusionLayer → 64D embedding
Search: FAISS IndexFlatL2 (551K vectors, 64D)
```

### Model 2: Audio-Only Autoencoder

```
Input: 11 audio features → Encoder (11→32→16→8) → Latent (8D)
       Decoder (8→16→32→11) — reconstruction loss
       
Search: FAISS IndexFlatL2 (105K vectors, 8D)
```

The web application exposes both models via a **model selector** dropdown,
allowing users to switch between:
- "Model 1 (Hybrid)" — richer, lyrics-aware recommendations
- "Model 2 (Audio Only)" — faster, purely audio-characteristic-based

## Consequences

### Positive
- Each model is optimized for its specific input modality
- Model 2 can serve tracks that have no lyrics
- Users can compare recommendation quality between modalities
- Model 2 is significantly smaller (11KB vs ~440MB) and faster

### Negative
- Two separate FAISS indices must be maintained
- Two separate autocomplete data sources are needed
- Deployment requires downloading both model files on startup
- Code complexity increases (dual search paths in `app.py`)

### Risks
- Users may be confused by different results between models
- Model 2's 8D latent space may lose fine-grained distinctions
  (mitigated by the L2 distance switch — see ADR-002)
