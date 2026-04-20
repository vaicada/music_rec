"""
CLIP → Audio Bridge for Model 2 (AudioAutoencoder).

Pipeline:
  Image → CLIP (ImageMoodClassifier) → mood/context label
        → Audio Profile (9 features) → Z-score normalize
        → AudioAutoencoder encode → 32D embedding
        → FAISS search → top-k song recommendations

Audio Profile Table maps each mood/context to 9 audio feature values
  [energy, danceability, valence, tempo,
   acousticness, instrumentalness, speechiness, liveness, key]

Usage (CLI test):
    cd music_recommender
    python -m audio_model.clip_audio_bridge --test
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_model.config2 import CONFIG
from audio_model.model2 import AudioAutoencoder

# =============================================================================
# Audio Profiles
# Columns: energy, danceability, valence, tempo,
#          acousticness, instrumentalness, speechiness, liveness, key(0-1)
# =============================================================================

AUDIO_PROFILES: dict = {
    # Moods
    "Happy":     [0.75, 0.72, 0.85, 120.0, 0.15, 0.05, 0.10, 0.15, 0.45],
    "Sad":       [0.30, 0.38, 0.18,  82.0, 0.62, 0.10, 0.05, 0.08, 0.45],
    "Energetic": [0.92, 0.82, 0.75, 142.0, 0.05, 0.03, 0.08, 0.22, 0.45],
    "Calm":      [0.22, 0.42, 0.55,  88.0, 0.72, 0.22, 0.04, 0.10, 0.45],
    "Angry":     [0.88, 0.55, 0.15, 135.0, 0.08, 0.05, 0.15, 0.28, 0.45],
    # Contexts
    "Party":     [0.87, 0.88, 0.80, 126.0, 0.08, 0.03, 0.10, 0.30, 0.45],
    "Workout":   [0.92, 0.78, 0.70, 142.0, 0.05, 0.03, 0.08, 0.18, 0.45],
    "Study":     [0.28, 0.38, 0.50,  95.0, 0.68, 0.42, 0.04, 0.10, 0.45],
    "Relax":     [0.18, 0.42, 0.60,  84.0, 0.78, 0.28, 0.03, 0.10, 0.45],
    "Driving":   [0.72, 0.65, 0.65, 116.0, 0.18, 0.05, 0.06, 0.15, 0.45],
}


class CLIPAudioBridge:
    """
    Bridges CLIP image classification with Audio Autoencoder Model 2 FAISS search.
    Loads Model 2 + FAISS index once, then serves recommendations per image.
    """

    def __init__(
        self,
        model_path: str = CONFIG.model_path,
        stats_path: str = CONFIG.stats_path,
        faiss_path: str = CONFIG.faiss_index_path,
        mappings_path: str = CONFIG.faiss_mappings_path,
        feature_cols: Optional[List[str]] = None,
        embedding_dim: int = CONFIG.output_dim,
    ):
        self.feature_cols = feature_cols or CONFIG.audio_features
        self.embedding_dim = embedding_dim

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # Load normalization stats
        with open(stats_path) as f:
            stats = json.load(f)
        self._means = np.array(stats["mean"], dtype=np.float32)
        self._stds  = np.array(stats["std"],  dtype=np.float32)

        # Load AudioAutoencoder (Model 2)
        self.model = AudioAutoencoder(
            input_dim=len(self.feature_cols),
            latent_dim=embedding_dim,
            dropout=0.0,   # No dropout during inference
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Load FAISS index
        self.faiss_index = faiss.read_index(faiss_path)

        # Load song mappings
        with open(mappings_path, "rb") as f:
            self.mappings = pickle.load(f)

        print(f"[CLIPAudioBridge] Loaded AudioAutoencoder | FAISS: {self.faiss_index.ntotal} songs | Device: {device_str}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_artist(raw_artist: str) -> str:
        """Strip Python list brackets from artist strings.
        e.g. "['Ed Sheeran']" -> "Ed Sheeran"
             "['Ed Sheeran', 'Galantis']" -> "Ed Sheeran, Galantis"
        """
        s = str(raw_artist).strip()
        if s.startswith("[") and s.endswith("]"):
            # Remove outer brackets
            s = s[1:-1].strip()
            # Remove surrounding quotes from each item
            parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
            return ", ".join(p for p in parts if p)
        return s

    @staticmethod
    def _rescale_similarities(results: List[dict]) -> List[dict]:
        """Convert L2 distances to similarity scores (0-1 range).
        L2 distance: lower = more similar. We convert using min-max
        normalization: the closest result gets ~0.98, the farthest gets ~0.65.
        """
        if not results:
            return results
        dists = [r["similarity"] for r in results]
        max_dist = max(dists)
        min_dist = min(dists)
        spread = max_dist - min_dist
        if spread < 1e-10:
            for r in results:
                r["similarity"] = 0.95
        else:
            for r in results:
                # Invert: smaller distance → higher similarity
                normalized = 1.0 - (r["similarity"] - min_dist) / spread
                r["similarity"] = round(0.65 + normalized * 0.33, 4)
        return results

    def _normalize(self, raw: List[float]) -> np.ndarray:
        """Z-score normalize a raw audio feature vector."""
        arr = np.array(raw, dtype=np.float32)
        return (arr - self._means) / (self._stds + 1e-8)

    def _encode(self, norm_features: np.ndarray) -> np.ndarray:
        """Encode normalized features → 8D raw embedding via AudioAutoencoder."""
        tensor = torch.tensor(norm_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode(tensor)
        return emb.cpu().numpy().astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend_from_label(
        self,
        label: str,
        top_k: int = 10,
    ) -> List[dict]:
        """
        Given a mood/context label, return top-k song recommendations
        using the audio profile → AudioAutoencoder → FAISS pipeline.
        """
        label_key = label.capitalize()
        if label_key not in AUDIO_PROFILES:
            label_key = "Calm"   # Fallback

        raw_profile = AUDIO_PROFILES[label_key]
        norm_features = self._normalize(raw_profile)
        query_emb = self._encode(norm_features)   # shape: (1, 32)

        distances, indices = self.faiss_index.search(query_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.mappings):
                continue
            song_meta = self.mappings[idx]
            results.append({
                "song":       song_meta.get("song", "Unknown"),
                "artist":     self._clean_artist(song_meta.get("artist", "Unknown")),
                "genre":      song_meta.get("genre", "") or (song_meta.get("year", "") and f"Year: {song_meta['year']}") or "",
                "emotion":    song_meta.get("emotion", "") or song_meta.get("album", ""),
                "similarity": float(dist),
            })
        return self._rescale_similarities(results)

    def recommend_from_song(
        self,
        query: str,
        artist: Optional[str] = None,
        top_k: int = 10,
    ) -> Optional[dict]:
        """
        Finds a song by name (and artist), retrieves its Model 2 embedding from FAISS,
        and returns top-k similar songs based purely on Audio features.
        Returns None if song not found.
        """
        query_lower = str(query).lower()
        artist_lower = str(artist).lower() if artist else None

        best_idx = -1
        for idx, meta in enumerate(self.mappings):
            s_name = str(meta.get("song", "Unknown")).lower()
            a_name = str(meta.get("artist", "Unknown")).lower()

            # Exact match prioritization
            if query_lower == s_name:
                if not artist_lower or artist_lower in a_name:
                    best_idx = idx
                    break
            # Fallback substring match
            elif best_idx == -1 and query_lower in s_name:
                if not artist_lower or artist_lower in a_name:
                    best_idx = idx

        if best_idx == -1:
            return None

        # Get embedding directly from FAISS
        query_emb = self.faiss_index.reconstruct(best_idx)
        query_emb = np.expand_dims(query_emb, axis=0)

        # Search (top_k + 1 to exclude the query itself)
        distances, indices = self.faiss_index.search(query_emb, top_k + 1)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if int(idx) == best_idx or idx < 0 or idx >= len(self.mappings):
                continue
            if len(results) >= top_k:
                break
            song_meta = self.mappings[idx]
            results.append({
                "song":       str(song_meta.get("song", "Unknown")),
                "artist":     self._clean_artist(song_meta.get("artist", "Unknown")),
                "genre":      str(song_meta.get("genre", "")) or (song_meta.get("year", "") and f"Year: {song_meta['year']}") or "",
                "emotion":    str(song_meta.get("emotion", "")) or str(song_meta.get("album", "")),
                "similarity": float(dist),
            })

        # Rescale similarity scores to a meaningful range
        results = self._rescale_similarities(results)

        matched_song_meta = self.mappings[best_idx]
        return {
            "matched_song": {
                "song_name":   str(matched_song_meta.get("song", "Unknown")),
                "artist_name": self._clean_artist(matched_song_meta.get("artist", "Unknown")),
                "match_score": 100
            },
            "recommendations": results,
        }


# =============================================================================
# CLI test
# =============================================================================

def _test_bridge():
    print("[test] Initialising CLIPAudioBridge ...")
    bridge = CLIPAudioBridge()

    test_labels = ["Happy", "Sad", "Relax", "Workout", "Party"]
    for label in test_labels:
        print(f"\n--- {label} ---")
        recs = bridge.recommend_from_label(label, top_k=5)
        for r in recs:
            print(f"  [{r['similarity']:.4f}] {r['song']} - {r['artist']} ({r['genre']})")
    print("\n[test] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a quick CLI test")
    args = parser.parse_args()

    if args.test:
        _test_bridge()
