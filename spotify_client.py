"""
Spotify API Client Module for Music Recommender.

This module handles all interactions with the Spotify Web API:
- Authentication (Client Credentials Flow - no user login needed)
- Track search
- Audio features retrieval
- Audio analysis retrieval (for Model 2 enrichment)

Usage:
    from spotify_client import SpotifyClient
    
    client = SpotifyClient()  # Auto-loads from .env
    
    # Search for a track
    track = client.search_track("Shape of You", "Ed Sheeran")
    
    # Get audio features
    features = client.get_audio_features(track['id'])
    
    # Get audio analysis (detailed timbre/pitch)
    analysis = client.get_audio_analysis(track['id'])

Author: Graduation Project
Created: 2026-03-12
"""

import os
import time
import base64
import requests
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class SpotifyClient:
    """
    Spotify Web API client using Client Credentials Flow.
    
    This flow is server-to-server and does NOT require user login.
    It provides access to all public Spotify data including:
    - Search, Track info, Audio Features, Audio Analysis, Recommendations
    """
    
    AUTH_URL = "https://accounts.spotify.com/api/token"
    BASE_URL = "https://api.spotify.com/v1"
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Spotify credentials not found. "
                "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file."
            )
        
        self._token = None
        self._token_expires_at = 0
    
    # =========================================================================
    # Authentication
    # =========================================================================
    
    def _get_token(self) -> str:
        """Get a valid access token, refreshing if expired."""
        if self._token and time.time() < self._token_expires_at:
            return self._token
        
        # Request new token
        auth_str = f"{self.client_id}:{self.client_secret}"
        auth_b64 = base64.b64encode(auth_str.encode()).decode()
        
        response = requests.post(
            self.AUTH_URL,
            headers={"Authorization": f"Basic {auth_b64}"},
            data={"grant_type": "client_credentials"},
            timeout=10
        )
        
        if response.status_code != 200:
            raise ConnectionError(
                f"Spotify auth failed: {response.status_code} - {response.text}"
            )
        
        data = response.json()
        self._token = data["access_token"]
        # Expire 60 seconds early to avoid edge cases
        self._token_expires_at = time.time() + data["expires_in"] - 60
        
        print(f"[Spotify] Access token obtained (expires in {data['expires_in']}s)")
        return self._token
    
    def _headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        return {"Authorization": f"Bearer {self._get_token()}"}
    
    def _get(self, endpoint: str, params: dict = None) -> Dict:
        """Make authenticated GET request to Spotify API."""
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, headers=self._headers(), params=params, timeout=15)
        
        if response.status_code == 429:
            # Rate limited - wait and retry
            retry_after = int(response.headers.get("Retry-After", 5))
            print(f"[Spotify] Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
            return self._get(endpoint, params)
        
        if response.status_code != 200:
            print(f"[Spotify] API error: {response.status_code} - {response.text[:200]}")
            return None
        
        return response.json()
    
    # =========================================================================
    # Search
    # =========================================================================
    
    def search_track(self, song_name: str, artist_name: str = None) -> Optional[Dict]:
        """
        Search for a track on Spotify.
        
        Returns the best matching track with: id, name, artists, album,
        popularity, preview_url, album art.
        """
        query = song_name
        if artist_name:
            query = f"track:{song_name} artist:{artist_name}"
        
        data = self._get("search", params={
            "q": query,
            "type": "track",
            "limit": 1
        })
        
        if not data or not data.get("tracks", {}).get("items"):
            return None
        
        track = data["tracks"]["items"][0]
        
        return {
            "id": track["id"],
            "name": track.get("name", ""),
            "artists": [a["name"] for a in track.get("artists", [])],
            "artist": track["artists"][0]["name"] if track.get("artists") else "Unknown",
            "album": track.get("album", {}).get("name", ""),
            "album_art": track.get("album", {}).get("images", [{}])[0].get("url") if track.get("album", {}).get("images") else None,
            "popularity": track.get("popularity", 0),
            "preview_url": track.get("preview_url"),
            "spotify_url": track.get("external_urls", {}).get("spotify"),
            "duration_ms": track.get("duration_ms", 0),
            "explicit": track.get("explicit", False),
        }
    
    # =========================================================================
    # Audio Features
    # =========================================================================
    
    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        """
        Get Spotify audio features for a track.
        
        Returns features matching our model's input:
        energy, danceability, valence, tempo, acousticness,
        instrumentalness, speechiness, liveness, key, mode, loudness
        """
        data = self._get(f"audio-features/{track_id}")
        
        if not data:
            return None
        
        return {
            "energy": data.get("energy"),
            "danceability": data.get("danceability"),
            "valence": data.get("valence"),
            "tempo": data.get("tempo"),
            "acousticness": data.get("acousticness"),
            "instrumentalness": data.get("instrumentalness"),
            "speechiness": data.get("speechiness"),
            "liveness": data.get("liveness"),
            "key": data.get("key"),
            "mode": data.get("mode"),
            "loudness": data.get("loudness"),
            "duration_ms": data.get("duration_ms"),
            "time_signature": data.get("time_signature"),
        }
    
    def get_audio_features_for_model(self, track_id: str) -> Optional[List[float]]:
        """
        Get audio features formatted for Model 1 input (9 features).
        
        Returns list in order: [energy, danceability, valence, tempo,
        acousticness, instrumentalness, speechiness, liveness, key]
        
        Note: Features are in Spotify API scale (0-1 for most, BPM for tempo, 0-11 for key)
        """
        features = self.get_audio_features(track_id)
        if not features:
            return None
        
        return [
            features["energy"],
            features["danceability"],
            features["valence"],
            features["tempo"],
            features["acousticness"],
            features["instrumentalness"],
            features["speechiness"],
            features["liveness"],
            features["key"] / 11.0,  # Normalize key to 0-1
        ]
    
    def get_batch_audio_features(self, track_ids: List[str]) -> List[Optional[Dict]]:
        """
        Get audio features for multiple tracks (max 100 per request).
        More efficient than calling get_audio_features() in a loop.
        """
        results = []
        
        # Process in batches of 100 (Spotify's limit)
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i + 100]
            ids_str = ",".join(batch)
            
            data = self._get("audio-features", params={"ids": ids_str})
            
            if data and "audio_features" in data:
                results.extend(data["audio_features"])
            else:
                results.extend([None] * len(batch))
        
        return results
    
    # =========================================================================
    # Audio Analysis (for Model 2 enrichment)
    # =========================================================================
    
    def get_audio_analysis(self, track_id: str) -> Optional[Dict]:
        """
        Get detailed audio analysis including timbre and pitch vectors.
        
        Returns aggregated segment-level data useful for Model 2:
        - timbre_mean[12]: average timbre vector
        - timbre_std[12]: timbre variation
        - pitch_mean[12]: average pitch vector  
        - section_count: number of sections
        - avg_segment_loudness: average loudness of segments
        - tempo_confidence: how confident Spotify is about tempo
        """
        data = self._get(f"audio-analysis/{track_id}")
        
        if not data:
            return None
        
        import numpy as np
        
        # Extract segment-level timbre and pitch
        segments = data.get("segments", [])
        
        if not segments:
            return None
        
        timbres = [s["timbre"] for s in segments if "timbre" in s]
        pitches = [s["pitches"] for s in segments if "pitches" in s]
        loudnesses = [s["loudness_max"] for s in segments if "loudness_max" in s]
        
        timbre_array = np.array(timbres)
        pitch_array = np.array(pitches)
        
        # Aggregate to track-level features
        result = {
            "timbre_mean": timbre_array.mean(axis=0).tolist(),  # [12]
            "timbre_std": timbre_array.std(axis=0).tolist(),    # [12]
            "pitch_mean": pitch_array.mean(axis=0).tolist(),    # [12]
            "pitch_std": pitch_array.std(axis=0).tolist(),      # [12]
            "section_count": len(data.get("sections", [])),
            "avg_segment_loudness": float(np.mean(loudnesses)) if loudnesses else 0,
            "tempo_confidence": data.get("track", {}).get("tempo_confidence", 0),
            "key_confidence": data.get("track", {}).get("key_confidence", 0),
            "mode_confidence": data.get("track", {}).get("mode_confidence", 0),
            "time_signature_confidence": data.get("track", {}).get("time_signature_confidence", 0),
            "num_segments": len(segments),
            "num_bars": len(data.get("bars", [])),
            "num_beats": len(data.get("beats", [])),
        }
        
        return result
    
    # =========================================================================
    # Recommendations
    # =========================================================================
    
    def get_recommendations(
        self,
        seed_tracks: List[str] = None,
        seed_artists: List[str] = None,
        seed_genres: List[str] = None,
        limit: int = 10,
        **target_attributes
    ) -> Optional[List[Dict]]:
        """
        Get Spotify recommendations based on seed tracks/artists/genres.
        
        Can also specify target audio attributes:
        target_energy, target_danceability, target_valence, etc.
        """
        params = {"limit": limit}
        
        if seed_tracks:
            params["seed_tracks"] = ",".join(seed_tracks[:5])  # Max 5
        if seed_artists:
            params["seed_artists"] = ",".join(seed_artists[:5])
        if seed_genres:
            params["seed_genres"] = ",".join(seed_genres[:5])
        
        # Add target attributes (e.g., target_energy=0.8)
        for key, value in target_attributes.items():
            if key.startswith("target_") or key.startswith("min_") or key.startswith("max_"):
                params[key] = value
        
        data = self._get("recommendations", params=params)
        
        if not data or not data.get("tracks"):
            return None
        
        return [
            {
                "id": t["id"],
                "name": t["name"],
                "artist": t["artists"][0]["name"],
                "album": t["album"]["name"],
                "album_art": t["album"]["images"][0]["url"] if t["album"]["images"] else None,
                "popularity": t["popularity"],
                "spotify_url": t["external_urls"].get("spotify"),
            }
            for t in data["tracks"]
        ]
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    def test_connection(self) -> bool:
        """Test if Spotify API connection works."""
        try:
            token = self._get_token()
            if token:
                print("[Spotify] ✅ Connection successful!")
                
                # Quick test search
                result = self.search_track("Bohemian Rhapsody", "Queen")
                if result:
                    print(f"[Spotify] ✅ Search works: {result['name']} by {result['artist']}")
                    print(f"[Spotify]    Popularity: {result['popularity']}, Album: {result['album']}")
                    if result.get('album_art'):
                        print(f"[Spotify]    Album Art: {result['album_art'][:60]}...")
                    
                    # Test audio features (may be deprecated for new apps)
                    features = self.get_audio_features(result['id'])
                    if features:
                        print(f"[Spotify] ✅ Audio Features works: energy={features['energy']}, "
                              f"danceability={features['danceability']}")
                    else:
                        print(f"[Spotify] ⚠️  Audio Features returned 403 - this endpoint is")
                        print(f"[Spotify]    DEPRECATED for new apps since Nov 2024.")
                        print(f"[Spotify]    Search, Track Info, and Recommendations still work.")
                
                return True
        except Exception as e:
            print(f"[Spotify] ❌ Connection failed: {e}")
            return False


# =============================================================================
# Quick test when run directly
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Spotify API Connection")
    print("=" * 50)
    
    client = SpotifyClient()
    client.test_connection()
