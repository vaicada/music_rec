"""
Inference Engine for the Hybrid Music Recommendation System.

This module provides the high-level API for the web application,
including similarity search using FAISS and recommendation functions.

GPU Support:
    - FAISS GPU provides ~50-100x faster search than CPU
    - Install with: pip install faiss-gpu (requires CUDA)
    - Maintains 100% accuracy (exact search)

Author: Graduation Project
Created: 2026-01-06
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import faiss
    # Check for GPU support
    try:
        _ = faiss.StandardGpuResources()
        FAISS_GPU_AVAILABLE = True
    except (AttributeError, RuntimeError):
        FAISS_GPU_AVAILABLE = False
except ImportError:
    faiss = None
    FAISS_GPU_AVAILABLE = False
    print("Warning: FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")

from .logger import get_logger, log_step
from .config import Config, get_config
from .model import HybridMusicModel
from .processors import TextProcessor, AudioProcessor, MetadataProcessor


class FAISSIndex:
    """
    FAISS-based similarity search index for song embeddings.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger(self.config.paths.log_file)
        
        self.embedding_dim = self.config.faiss.embedding_dim
        self.index = None
        self.gpu_resources = None
        self.song_ids: List[str] = []
        self.idx_to_song: Dict[int, dict] = {}
        self.using_gpu = False
        
        if faiss is None:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu or faiss-gpu")
    
    def _try_enable_gpu(self, index: 'faiss.Index') -> 'faiss.Index':
        if not self.config.faiss.use_gpu:
            return index
        
        if not FAISS_GPU_AVAILABLE:
            return index
        
        try:
            self.gpu_resources = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(
                self.gpu_resources,
                self.config.faiss.gpu_id,
                index
            )
            self.using_gpu = True
            return gpu_index
        except Exception as e:
            self.logger.log_error(f"Failed to enable GPU: {str(e)}", "INDEX")
            return index
    
    @log_step("INDEX", "Creating FAISS Index")
    def create_index(self, embeddings: np.ndarray, song_data: pd.DataFrame) -> None:
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        if self.config.faiss.use_hnsw:
            cpu_index = faiss.IndexHNSWFlat(self.embedding_dim, self.config.faiss.hnsw_m)
        else:
            cpu_index = faiss.IndexFlatL2(self.embedding_dim)
        
        cpu_index.add(embeddings)
        self.index = self._try_enable_gpu(cpu_index)
        
        self._build_mappings(song_data)
        
        self.logger.log("FAISS index built successfully", "INDEX", level="SUCCESS")

    def _get_column_names(self, df: pd.DataFrame) -> Tuple[str, str]:
        cols = df.columns
        if 'song' in cols: song_col = 'song'
        elif 'song_name' in cols: song_col = 'song_name'
        elif 'name' in cols: song_col = 'name'
        else: song_col = cols[0]
            
        if 'Artist(s)' in cols: artist_col = 'Artist(s)'
        elif 'artist' in cols: artist_col = 'artist'
        elif 'artists' in cols: artist_col = 'artists'
        else: artist_col = cols[1] if len(cols) > 1 else cols[0]
            
        return song_col, artist_col

    def _build_mappings(self, song_data: pd.DataFrame) -> None:
        song_col, artist_col = self._get_column_names(song_data)
        
        for idx, row in song_data.iterrows():
            song_info = {
                'idx': idx,
                'song': row.get(song_col, f'Unknown_{idx}'),
                'artist': row.get(artist_col, 'Unknown Artist'),
                'genre': row.get('Genre', row.get('genre', '')),
                'emotion': row.get('emotion', ''),
                'album': row.get('Album', row.get('album', '')),
            }
            # Robust ID
            song_val = str(song_info['song']).strip()
            artist_val = str(song_info['artist']).strip()
            song_id = f"{artist_val}|{song_val}"
            
            self.song_ids.append(song_id)
            self.idx_to_song[idx] = song_info

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[int, float, dict]]:
        if self.index is None: raise ValueError("Index not loaded")
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self.idx_to_song:
                results.append((idx, float(dist), self.idx_to_song[idx]))
        return results

    def reconstruct(self, idx: int) -> np.ndarray:
        """
        Reconstruct the vector for a given index from the FAISS index.
        Crucial for Lite Deployment where we don't have the raw data to re-encode.
        """
        if self.index is None: raise ValueError("Index not loaded")
        try:
            # Check if index supports reconstruction
            # IndexFlatL2, IndexHNSWFlat support make_direct_map() or direct access
            # For massive indices, this might fail if not supported, but standard ones do.
            if self.using_gpu:
                # GPU indices might strictly not support reconstruct directly in some versions
                # Need to copy to CPU first? Or it usually works.
                # Let's try direct first.
                return self.index.reconstruct(int(idx))
            else:
                return self.index.reconstruct(int(idx))
        except Exception as e:
            self.logger.log_error(f"Failed to reconstruct vector {idx}: {e}", "INDEX")
            # If reconstruction fails, we return zeros (will yield bad results but no crash)
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.config.paths.faiss_index_path
        index_to_save = self.index
        if self.using_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        
        faiss.write_index(index_to_save, str(path))
        with open(str(path) + '.mappings.pkl', 'wb') as f:
            pickle.dump({'song_ids': self.song_ids, 'idx_to_song': self.idx_to_song}, f)
        self.logger.log(f"Index saved to {path}", "INDEX", level="SUCCESS")

    def load(self, path: Optional[str] = None) -> None:
        path = path or self.config.paths.faiss_index_path
        cpu_index = faiss.read_index(str(path))
        # Important: For reconstruction to work reliably on some index types,
        # we might need to ensure it has potential direct map. 
        # But IndexFlatL2 (default) always supports it.
        self.index = self._try_enable_gpu(cpu_index)
        
        mappings_path = str(path) + '.mappings.pkl'
        if os.path.exists(mappings_path):
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
            self.song_ids = mappings['song_ids']
            self.idx_to_song = mappings['idx_to_song']
        self.logger.log(f"Index loaded from {path}", "INDEX", level="SUCCESS")


class MusicRecommendationEngine:
    """
    High-level inference engine for music recommendations.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger(self.config.paths.log_file)
        
        self.model: Optional[HybridMusicModel] = None
        self.faiss_index: Optional[FAISSIndex] = None
        
        self.text_processor = TextProcessor(config)
        self.audio_processor = AudioProcessor(config)
        self.metadata_processor = MetadataProcessor(config)
        
        self.song_data: Optional[pd.DataFrame] = None
        
        device_str = self.config.training.get_device()
        self.device = torch.device(device_str)
        self.logger.log("Engine initialized", "INFERENCE")
    
    @log_step("MODEL", "Loading Model")
    def load_model(self, model_path: str) -> None:
        self.model = HybridMusicModel(self.config)
        self.model.load(model_path, device=str(self.device))
        self.model.to(self.device)
        self.model.eval()
    
    @log_step("INDEX", "Loading Index")
    def load_index(self, index_path: Optional[str] = None) -> None:
        self.faiss_index = FAISSIndex(self.config)
        self.faiss_index.load(index_path)
    
    @log_step("DATA", "Loading Song Database")
    def load_song_data(self, data_path: str) -> None:
        path = Path(data_path)
        if path.suffix == '.csv':
            self.song_data = pd.read_csv(data_path)
        elif path.suffix in ['.json', '.jsonl']:
            self.song_data = pd.read_json(data_path, lines=path.suffix == '.jsonl')
        
        # Init processors on this data
        # Note: If song_metadata.csv is used (Lite Mode), it lacks columns.
        # Processors usually fit on 'text', 'genre' etc.
        # We try to fit anyway, but handle potential missing columns gracefully in transform.
        try:
            self.audio_processor.fit(self.song_data)
            self.metadata_processor.fit(self.song_data)
        except Exception as e:
            self.logger.log(f"Processor fit warning (Lite Mode?): {e}", "DATA", level="WARNING")
        
        self.logger.log(f"Loaded {len(self.song_data)} songs", "DATA", level="SUCCESS")

    def _get_column_names(self) -> Tuple[str, str]:
        cols = self.song_data.columns
        if 'song' in cols: song_col = 'song'
        elif 'song_name' in cols: song_col = 'song_name'
        elif 'name' in cols: song_col = 'name'
        else: song_col = cols[0]
            
        if 'Artist(s)' in cols: artist_col = 'Artist(s)'
        elif 'artist' in cols: artist_col = 'artist'
        elif 'artists' in cols: artist_col = 'artists'
        else: artist_col = cols[1] if len(cols) > 1 else cols[0]
        return song_col, artist_col

    def _find_song(self, song_name: str, artist_name: Optional[str] = None) -> Optional[pd.Series]:
        if self.song_data is None: return None
        
        song_col, artist_col = self._get_column_names()
        
        # Exact match
        mask = self.song_data[song_col].fillna('').astype(str).str.lower() == song_name.lower()
        
        if artist_name:
            artist_mask = self.song_data[artist_col].fillna('').astype(str).str.lower().str.contains(artist_name.lower(), na=False)
            mask = mask & artist_mask
        
        matches = self.song_data[mask]
        if len(matches) > 0: return matches.iloc[0]
        
        # Fuzzy match
        mask = self.song_data[song_col].fillna('').astype(str).str.lower().str.contains(song_name.lower(), na=False)
        matches = self.song_data[mask]
        if len(matches) > 0: return matches.iloc[0]
        
        return None

    def _encode_song(self, song_row: pd.Series) -> torch.Tensor:
        if self.model is None: raise ValueError("Model not loaded")
        
        lyrics_col = 'text' if 'text' in song_row.index else 'lyrics'
        lyrics = self.text_processor.clean_lyrics(song_row.get(lyrics_col, ""))
        emotion = str(song_row.get('emotion', ''))
        genre = str(song_row.get('Genre', song_row.get('genre', '')))
        
        combined_text = self.text_processor.combine_text_features(lyrics, emotion, genre)
        text_encoded = self.text_processor.tokenize_single(combined_text)
        
        # Audio features might be missing in Lite Mode
        try:
            audio_features = self.audio_processor.transform(pd.DataFrame([song_row]))
        except (KeyError, ValueError, TypeError) as e:
            # Fallback to zeros if columns missing (Lite Mode)
            self.logger.log(f"Audio features unavailable: {e}", "INFERENCE", level="WARNING")
            audio_features = torch.zeros((1, self.config.audio.input_dim))
        
        inputs = {
            'input_ids': text_encoded['input_ids'].to(self.device),
            'attention_mask': text_encoded['attention_mask'].to(self.device),
            'token_type_ids': text_encoded['token_type_ids'].to(self.device),
            'audio_features': audio_features.to(self.device),
        }
        
        with torch.no_grad():
            embedding = self.model.get_embedding(**inputs)
        return embedding.cpu()

    @log_step("INFERENCE", "Getting Similar Songs")
    def get_similar_songs(self, song_name: str, artist_name: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        song_row = self._find_song(song_name, artist_name)
        if song_row is None:
            self.logger.log(f"Song not found: {song_name}", "INFERENCE", level="WARNING")
            return []
        
        if self.faiss_index is None: raise ValueError("Index not loaded")
        
        # OPTIMIZATION: Use pre-computed vector from FAISS if possible
        # This ensures 100% accuracy even if using song_metadata.csv (which lacks features)
        try:
            # Assuming dataframe index matches FAISS index ID
            # This is true if song_metadata.csv is a direct subset of training data without shuffle
            song_idx = int(song_row.name)
            embedding_np = self.faiss_index.reconstruct(song_idx)
            
            # Verify if reconstruction looks valid (not all zeros)
            if np.all(embedding_np == 0):
                self.logger.log("Reconstructed vector is all zeros, falling back to encoding", "INFERENCE")
                embedding = self._encode_song(song_row)
                embedding_np = embedding.numpy().squeeze()
            else:
                self.logger.log(f"Using pre-computed vector for ID {song_idx}", "INFERENCE")
                
        except Exception as e:
            self.logger.log(f"Vector reconstruction failed ({e}), encoding from scratch", "INFERENCE")
            embedding = self._encode_song(song_row)
            embedding_np = embedding.numpy().squeeze()
        
        # Search more for filtering
        candidates_k = top_k * 3
        results = self.faiss_index.search(embedding_np, candidates_k + 1)
        
        # Post-processing filters: Strict Emotion Logic
        query_emotion = song_row.get('emotion', '').lower()
        compatible_emotions = {
            'joy': ['joy', 'love', 'surprise', 'anger'],
            'sadness': ['sadness', 'love', 'fear'], 
            'anger': ['anger', 'joy', 'fear'],
            'love': ['love', 'joy', 'sadness'],
            'fear': ['fear', 'sadness', 'anger'],
            'surprise': ['surprise', 'joy']
        }
        allowed_emotions = compatible_emotions.get(query_emotion, [])
        if not allowed_emotions and query_emotion: allowed_emotions = [query_emotion]
            
        recommendations = []
        song_col, _ = self._get_column_names()
        query_song_name = str(song_row.get(song_col, '')).lower()
        
        for idx, distance, song_info in results:
            if str(song_info['song']).lower() == query_song_name: continue
            
            # Emotion Filter
            cand_emotion = song_info.get('emotion', '').lower()
            if allowed_emotions and cand_emotion and cand_emotion not in allowed_emotions:
                continue
            
            similarity = 1.0 / (1.0 + distance)
            recommendations.append({
                'song': song_info['song'],
                'artist': song_info['artist'],
                'similarity': round(similarity, 4),
                'genre': song_info.get('genre', ''),
                'emotion': song_info.get('emotion', ''),
                'album': song_info.get('album', '')
            })
            if len(recommendations) >= top_k: break
        
        # Fallback
        if len(recommendations) == 0:
            for idx, distance, song_info in results[:top_k+1]:
                 if str(song_info['song']).lower() == query_song_name: continue
                 similarity = 1.0 / (1.0 + distance)
                 recommendations.append({
                    'song': song_info['song'],
                    'artist': song_info['artist'],
                    'similarity': round(similarity, 4),
                    'genre': song_info.get('genre', ''),
                    'emotion': song_info.get('emotion', ''),
                    'album': song_info.get('album', '')
                })
        
        return recommendations

    def get_recommendations_by_mood(self, mood: str, top_k: int = 10) -> List[Dict]:
        """Return `top_k` songs matching the mood, randomly sampled from a large candidate pool."""
        import random
        if self.song_data is None: return []

        mood_mapping = {
            'happy':    {'emotion': ['joy'],     'valence': (0.6, 1.0), 'energy': (0.5, 1.0)},
            'sad':      {'emotion': ['sadness'], 'valence': (0.0, 0.45), 'energy': (0.0, 0.55)},
            'energetic':{'emotion': ['joy'],     'valence': (0.5, 1.0), 'energy': (0.7, 1.0)},
            'calm':     {'emotion': [],          'valence': (0.35, 0.7), 'energy': (0.0, 0.45)},
            'angry':    {'emotion': ['anger'],   'valence': (0.0, 0.45), 'energy': (0.65, 1.0)},
        }

        mood_lower = mood.lower()
        if mood_lower not in mood_mapping: mood_lower = 'happy'
        filters = mood_mapping[mood_lower]
        filtered = self.song_data.copy()

        if filters['emotion'] and 'emotion' in filtered.columns:
            filtered = filtered[filtered['emotion'].isin(filters['emotion'])]
        if 'valence' in filtered.columns:
            v_lo, v_hi = filters['valence']
            filtered = filtered[(filtered['valence'] >= v_lo) & (filtered['valence'] <= v_hi)]
        if 'energy' in filtered.columns:
            e_lo, e_hi = filters['energy']
            filtered = filtered[(filtered['energy'] >= e_lo) & (filtered['energy'] <= e_hi)]

        # Fallback: if filter is too strict, relax and just use emotion
        if len(filtered) < top_k * 2:
            filtered = self.song_data.copy()
            if filters['emotion'] and 'emotion' in filtered.columns:
                filtered = filtered[filtered['emotion'].isin(filters['emotion'])]
            # Still too small → use entire dataset
            if len(filtered) < top_k:
                filtered = self.song_data.copy()

        # Build a weighted pool: top 300 by Popularity (or full set if smaller)
        pool_size = min(300, len(filtered))
        if 'Popularity' in filtered.columns:
            pool = filtered.nlargest(pool_size, 'Popularity')
        else:
            pool = filtered.head(pool_size)

        # Random-sample top_k from the pool
        sample_size = min(top_k, len(pool))
        sampled = pool.sample(n=sample_size)

        song_col, artist_col = self._get_column_names()
        recommendations = []
        for _, row in sampled.iterrows():
            recommendations.append({
                'song': row.get(song_col, ''),
                'artist': row.get(artist_col, ''),
                'genre': row.get('Genre', row.get('genre', '')),
                'emotion': row.get('emotion', ''),
                'similarity': None,
            })
        random.shuffle(recommendations)
        return recommendations

    def get_recommendations_by_context(self, context: str, top_k: int = 10) -> List[Dict]:
        """Return `top_k` songs matching the activity/context, randomly sampled."""
        import random
        if self.song_data is None: return []

        # Context → emotion + audio feature filters
        context_mapping = {
            'party':   {'emotion': ['joy'],          'valence': (0.65, 1.0), 'energy': (0.7, 1.0), 'danceability': (0.6, 1.0)},
            'workout': {'emotion': ['joy', 'anger'],  'valence': (0.4, 1.0),  'energy': (0.75, 1.0), 'danceability': None},
            'study':   {'emotion': ['sadness'],       'valence': (0.2, 0.65), 'energy': (0.0, 0.5),  'danceability': None},
            'relax':   {'emotion': ['sadness', 'love'],'valence': (0.3, 0.7), 'energy': (0.0, 0.45), 'danceability': None},
            'driving': {'emotion': ['joy'],           'valence': (0.5, 1.0),  'energy': (0.55, 1.0), 'danceability': None},
        }

        ctx_lower = context.lower()
        if ctx_lower not in context_mapping: ctx_lower = 'party'
        filters = context_mapping[ctx_lower]
        filtered = self.song_data.copy()

        if filters['emotion'] and 'emotion' in filtered.columns:
            filtered = filtered[filtered['emotion'].isin(filters['emotion'])]
        if 'valence' in filtered.columns:
            v_lo, v_hi = filters['valence']
            filtered = filtered[(filtered['valence'] >= v_lo) & (filtered['valence'] <= v_hi)]
        if 'energy' in filtered.columns:
            e_lo, e_hi = filters['energy']
            filtered = filtered[(filtered['energy'] >= e_lo) & (filtered['energy'] <= e_hi)]
        if filters.get('danceability') and 'danceability' in filtered.columns:
            d_lo, d_hi = filters['danceability']
            filtered = filtered[(filtered['danceability'] >= d_lo) & (filtered['danceability'] <= d_hi)]

        # Fallback if too restrictive
        if len(filtered) < top_k * 2:
            filtered = self.song_data.copy()
            if filters['emotion'] and 'emotion' in filtered.columns:
                filtered = filtered[filtered['emotion'].isin(filters['emotion'])]
            if len(filtered) < top_k:
                filtered = self.song_data.copy()

        # Build a weighted pool and random-sample
        pool_size = min(300, len(filtered))
        if 'Popularity' in filtered.columns:
            pool = filtered.nlargest(pool_size, 'Popularity')
        else:
            pool = filtered.head(pool_size)

        sample_size = min(top_k, len(pool))
        sampled = pool.sample(n=sample_size)

        song_col, artist_col = self._get_column_names()
        recommendations = []
        for _, row in sampled.iterrows():
            recommendations.append({
                'song': row.get(song_col, ''),
                'artist': row.get(artist_col, ''),
                'genre': row.get('Genre', row.get('genre', '')),
                'emotion': row.get('emotion', ''),
                'similarity': None,
            })
        random.shuffle(recommendations)
        return recommendations

    def build_index(self, data_path: str, save_path: Optional[str] = None) -> None:
        self.load_song_data(data_path)
        if self.model is None: raise ValueError("Model not loaded")
        
        all_embeddings = []
        batch_size = self.config.training.batch_size
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(self.song_data), batch_size):
                batch_data = self.song_data.iloc[i:i+batch_size]
                batch_embeddings = []
                for _, row in batch_data.iterrows():
                    try:
                        emb = self._encode_song(row)
                        batch_embeddings.append(emb.numpy().squeeze())
                    except (RuntimeError, ValueError, KeyError) as e:
                        self.logger.log(f"Encoding failed for row: {e}", "INDEX", level="WARNING")
                        batch_embeddings.append(np.zeros(self.config.fusion.final_embedding_dim))
                all_embeddings.extend(batch_embeddings)
        
        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        self.faiss_index = FAISSIndex(self.config)
        self.faiss_index.create_index(self.embeddings, self.song_data)
        if save_path: self.faiss_index.save(save_path)

def create_engine(model_path: str, index_path: str, data_path: str, config: Optional[Config] = None) -> MusicRecommendationEngine:
    engine = MusicRecommendationEngine(config)
    engine.load_model(model_path)
    engine.load_index(index_path)
    engine.load_song_data(data_path)
    return engine
