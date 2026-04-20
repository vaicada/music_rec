# ADR-002: L2 Distance for Model 2 (Replacing Cosine Similarity)

## Status
Accepted

## Date
2026-04-04

## Context

Model 2 (Audio-Only Autoencoder) originally used **cosine similarity** for
FAISS search with a 32-dimensional latent space. During evaluation, we observed
a critical problem:

**All similarity scores clustered at 99.9%.**

```
Query: "Shape of You" (Ed Sheeran)
Result 1: "Blinding Lights"  → similarity: 99.94%
Result 2: "Bad Guy"          → similarity: 99.92%
Result 3: "Someone Like You" → similarity: 99.91%
```

This happened because:
1. Cosine similarity measures angle, not magnitude
2. Autoencoder latent vectors tend to occupy a narrow cone in high-D space
3. With 32 dimensions, angular differences between any two vectors were tiny
4. Users had no way to distinguish "very similar" from "somewhat similar"

## Decision

We made two simultaneous changes:

### 1. Switch from Cosine Similarity to L2 (Euclidean) Distance
- Replace `faiss.IndexFlatIP` (inner product) with `faiss.IndexFlatL2`
- L2 measures absolute distance, not angle
- Similarity score: `1 / (1 + distance)` — provides a 0-to-1 range

### 2. Reduce Latent Dimension from 32 to 8
- Encoder: 11 → 32 → 16 → **8** (was 32)
- Creates a tighter information bottleneck
- Forces the autoencoder to learn only the most discriminative features
- Retrained model is only ~11KB

## Consequences

### Positive
- Similarity scores now have **meaningful dynamic range**: 40%-95%
- Users can clearly see which songs are "very similar" vs "somewhat similar"
- Smaller latent dimension → smaller FAISS index → faster search
- Model file is tiny (11KB) — can be committed directly to Git

### Negative
- Required full retraining of Model 2
- Previous Model 2 indices are incompatible (dimension mismatch)
- 8D may lose some fine-grained audio distinctions
  (acceptable trade-off for better score interpretability)

### Metrics After Change
```
Query: "Shape of You" (Ed Sheeran)
Result 1: "Cheap Thrills"    → similarity: 89.23%  ✓ Clear ranking
Result 2: "Blinding Lights"  → similarity: 82.15%
Result 3: "Someone Like You" → similarity: 61.47%  ✓ Clearly different
```
