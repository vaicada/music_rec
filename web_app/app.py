"""
Music Recommender Web API - FastAPI Backend Application.

Serves the Music Recommendation Engine through RESTful endpoints.
Architecture: Frontend (HTML/CSS/JS) → FastAPI → MusicRecommendationEngine → PyTorch + FAISS

API Endpoints:
    GET  /                        : Main HTML page
    GET  /api/health              : Health check
    GET  /api/search              : Similar song search
    GET  /api/mood/{m}            : Mood-based recommendations
    GET  /api/youtube             : YouTube embed URL
    POST /api/recommend-from-image: Image-based recommendation (CLIP)
    POST /api/auth/register       : Register new user
    POST /api/auth/login          : Login → JWT token
    GET  /api/auth/me             : Get current user info
    GET  /api/history             : Get search history
    DELETE /api/history/{id}      : Delete history entry
    DELETE /api/history           : Clear all history
    POST /api/playlists           : Create playlist
    GET  /api/playlists           : List playlists
    GET  /api/playlists/{id}      : Get playlist with tracks
    PUT  /api/playlists/{id}      : Update playlist
    DELETE /api/playlists/{id}    : Delete playlist
    POST /api/playlists/{id}/tracks        : Add track
    DELETE /api/playlists/{id}/tracks/{tid}: Remove track

Usage: cd web_app && python -m uvicorn app:app --host 127.0.0.1 --port 8000
"""

from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List
from datetime import datetime
import asyncio
import traceback
import sys
import os
import time

# Load .env file BEFORE any module that reads environment variables (database, auth, etc.)
try:
    from dotenv import load_dotenv
    # Look for .env in project root (one level up from web_app/)
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_env_path)
    print(f"[OK] Loaded .env from {_env_path}")
except ImportError:
    print("[WARNING] python-dotenv not installed. Environment variables must be set manually.")

# Add parent directory to path to import hybrid_music_engine
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_web_app_dir  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)   # /app  → hybrid_music_engine, audio_model
sys.path.insert(1, _web_app_dir)    # /app/web_app → database, auth, download_helper

# Import download helper (try both relative and package paths)
try:
    from download_helper import ensure_files_ready
    ENABLE_AUTO_DOWNLOAD = True
except ImportError:
    try:
        from web_app.download_helper import ensure_files_ready
        ENABLE_AUTO_DOWNLOAD = True
    except ImportError:
        print("[WARNING] download_helper.py not found. Auto-download disabled.")
        ENABLE_AUTO_DOWNLOAD = False

from hybrid_music_engine import get_config
from hybrid_music_engine.inference import MusicRecommendationEngine

# Try to import yt_dlp, provide fallback if not available
try:
    import yt_dlp
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False
    print("[WARNING] yt-dlp not installed. YouTube features disabled.")


# Try to import ImageMoodClassifier for image-based recommendation
try:
    from hybrid_music_engine.image_processor import ImageMoodClassifier, is_clip_available
    IMAGE_PROCESSOR_AVAILABLE = is_clip_available()
except ImportError as e:
    ImageMoodClassifier = None  # type: ignore[assignment,misc]
    IMAGE_PROCESSOR_AVAILABLE = False
    print(f"[WARNING] ImageMoodClassifier import failed: {e}. Image-based recommendation disabled.")

# Try to import SpotifyClient for enriched search and album art
try:
    from spotify_client import SpotifyClient
    SPOTIFY_CLIENT_AVAILABLE = True
except ImportError as e:
    SpotifyClient = None  # type: ignore[assignment,misc]
    SPOTIFY_CLIENT_AVAILABLE = False
    print(f"[WARNING] SpotifyClient import failed: {e}. Spotify enrichment disabled.")

# Import database + auth modules (non-blocking — app still runs if DB fails)
try:
    from database import get_db, User, SearchHistory, Playlist, PlaylistTrack
    from auth import (
        get_current_user, get_optional_user,
        hash_password, verify_password,
        create_access_token, UserPublic,
    )
    from sqlalchemy.orm import Session
    DB_AVAILABLE = True
    print("[OK] Database + Auth modules loaded")
except Exception as db_import_err:
    DB_AVAILABLE = False
    print(f"[WARNING] DB/Auth import failed: {db_import_err}. Auth & playlist features disabled.")

    # ---------------------------------------------------------------------------
    # Fallback stubs — keep FastAPI from raising NameError at module load time
    # when database/auth modules are absent (e.g. Hugging Face Spaces deploy).
    # All real auth endpoints already guard with `if not DB_AVAILABLE` and return
    # HTTP 503, so these stubs are never actually called in production.
    # ---------------------------------------------------------------------------
    def get_db():  # type: ignore[misc]
        """Stub: DB unavailable — yields nothing, raises proper HTTP 503."""
        raise HTTPException(status_code=503, detail="Database not available. Auth & playlist features are disabled.")
        yield  # makes Python treat this as a generator (required for FastAPI Depends)

    def get_current_user():  # type: ignore[misc]
        """Stub: auth unavailable — raises HTTP 503 instead of crashing."""
        raise HTTPException(status_code=503, detail="Authentication not available. Database is not configured.")

    def get_optional_user():  # type: ignore[misc]
        return None

    def hash_password(pw: str) -> str:  # type: ignore[misc]
        return ""

    def verify_password(pw: str, hashed: str) -> bool:  # type: ignore[misc]
        return False

    def create_access_token(user_id: int, username: str) -> str:  # type: ignore[misc]
        return ""

    # Dummy model classes so type annotations don't break
    class User:  # type: ignore[no-redef]
        id: int
        username: str
        email: str

    class SearchHistory:  # type: ignore[no-redef]
        pass

    class Playlist:  # type: ignore[no-redef]
        pass

    class PlaylistTrack:  # type: ignore[no-redef]
        pass

    class Session:  # type: ignore[no-redef]
        pass

    class UserPublic:  # type: ignore[no-redef]
        pass

# Try to import CLIPAudioBridge for Model 2 image recommendations
try:
    from audio_model.clip_audio_bridge import CLIPAudioBridge
    AUDIO_BRIDGE_AVAILABLE = True
except ImportError as e:
    CLIPAudioBridge = None  # type: ignore[assignment,misc]
    AUDIO_BRIDGE_AVAILABLE = False
    print(f"[WARNING] CLIPAudioBridge import failed: {e}. Model 2 image recommendation disabled.")


# =============================================================================
# Utility Functions
# =============================================================================

def clean_artist_name(raw_artist: str) -> str:
    """Strip Python list brackets from artist strings.
    e.g. "['Ed Sheeran']" -> "Ed Sheeran"
         "['Ed Sheeran', 'Galantis']" -> "Ed Sheeran, Galantis"
    """
    s = str(raw_artist).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
        parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
        return ", ".join(p for p in parts if p)
    return s


# =============================================================================
# Pydantic Schemas
# =============================================================================

class SongResult(BaseModel):
    """Response model for a song result."""
    song: str
    artist: str
    genre: str
    emotion: str
    similarity: Optional[float] = None


class YouTubeResult(BaseModel):
    """Response model for YouTube video."""
    success: bool = True
    video_id: Optional[str] = None
    embed_url: str
    title: str = ""
    thumbnail: Optional[str] = None
    message: Optional[str] = None


class SearchResponse(BaseModel):
    """Response model for search results."""
    query: str
    results: List[SongResult]
    count: int


class ImageRecommendationResponse(BaseModel):
    """Response model for image-based recommendation."""
    success: bool
    detected_label: str = ""
    detected_type: str = ""
    confidence: float = 0.0
    recommendations: Optional[List[SongResult]] = None
    message: str = ""


# --- Auth Schemas ---

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        if len(v) > 50:
            raise ValueError("Username must be at most 50 characters")
        return v

    @field_validator("password")
    @classmethod
    def password_valid(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


# --- History Schemas ---

class HistoryItem(BaseModel):
    id: int
    query_song: str
    query_artist: str
    model_used: str
    results_count: int
    searched_at: datetime

    class Config:
        from_attributes = True


# --- Playlist Schemas ---

class PlaylistCreateRequest(BaseModel):
    name: str
    description: str = ""

    @field_validator("name")
    @classmethod
    def name_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Playlist name cannot be empty")
        if len(v) > 255:
            raise ValueError("Playlist name too long")
        return v


class TrackAddRequest(BaseModel):
    song_name: str
    artist_name: str = ""
    genre: str = ""
    emotion: str = ""
    spotify_url: str = ""
    album_art_url: str = ""


class TrackOut(BaseModel):
    id: int
    song_name: str
    artist_name: str
    genre: str
    emotion: str
    spotify_url: str
    album_art_url: str
    position: int
    added_at: datetime

    class Config:
        from_attributes = True


class PlaylistOut(BaseModel):
    id: int
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    track_count: int = 0

    class Config:
        from_attributes = True


class PlaylistDetailOut(BaseModel):
    id: int
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    tracks: List[TrackOut] = []

    class Config:
        from_attributes = True


# =============================================================================
# Global Engine Instance
# =============================================================================

engine: Optional[MusicRecommendationEngine] = None
image_classifier: Optional[ImageMoodClassifier] = None
spotify_client: Optional["SpotifyClient"] = None
audio_bridge: Optional["CLIPAudioBridge"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads the recommendation engine on startup.
    """
    global engine, image_classifier, spotify_client, audio_bridge
    print("\n" + "=" * 50)
    print("Starting Music Recommender API...")
    print("=" * 50)
    
    try:
        # STEP 0: Check and download missing files (for Vercel deployment)
        if ENABLE_AUTO_DOWNLOAD:
            print("\n[0/4] Checking required files...")
            ensure_files_ready()  # This will download missing files or exit if failed
        
        # Get configuration
        config = get_config()
        
        # Initialize engine
        print("[1/4] Initializing recommendation engine...")
        engine = MusicRecommendationEngine(config)
        
        # Load model
        model_path = os.path.join(config.paths.model_dir, "best_model.pth")
        print(f"[2/4] Loading model from {model_path}...")
        engine.load_model(model_path)
        
        # Load FAISS index
        print("[3/4] Loading FAISS index...")
        engine.load_index()
        
        # Load song data
        data_path = os.path.join(config.paths.processed_data_dir, "train.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(config.paths.model_dir, "song_metadata.csv")
        print(f"[4/4] Loading song data from {data_path}...")
        engine.load_song_data(data_path)
        
        print("\n[OK] Music Recommender API is ready!")
        print(f"[OK] API Docs available at: http://localhost:8000/docs")
        
        # Initialize image classifier (lazy load - model downloads on first use)
        if IMAGE_PROCESSOR_AVAILABLE:
            print("[INFO] Image-based recommendation is available (CLIP will load on first use)")
            image_classifier = ImageMoodClassifier()
        else:
            print("[INFO] Image-based recommendation is disabled (CLIP not available)")

        # Initialize Spotify client (non-blocking, graceful fallback)
        if SPOTIFY_CLIENT_AVAILABLE:
            try:
                spotify_client = SpotifyClient()
                # Quick token test (doesn't call data endpoints)
                spotify_client._get_token()
                print("[OK] Spotify API client initialized")
            except Exception as spotify_err:
                print(f"[WARNING] Spotify API init failed: {spotify_err}. Spotify features disabled.")
                spotify_client = None
        else:
            print("[INFO] Spotify client not available")

        # Initialize CLIPAudioBridge for Model 2 (optional - requires trained model)
        if AUDIO_BRIDGE_AVAILABLE:
            try:
                audio_bridge = CLIPAudioBridge()
                print("[OK] Model 2 (Audio-Only) CLIP bridge initialized")
            except Exception as m2_err:
                print(f"[INFO] Model 2 not ready yet: {m2_err}. Train first.")
                audio_bridge = None
        else:
            print("[INFO] Model 2 CLIP bridge not available")

        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize engine: {e}")
        print("[WARNING] API will start but recommendations won't work.")
        engine = None
    
    yield
    
    # Cleanup on shutdown
    print("\nShutting down Music Recommender API...")
    engine = None
    image_classifier = None
    spotify_client = None
    audio_bridge = None


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Music Recommender API",
    description="API for music recommendation based on hybrid deep learning model",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")




# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Music Recommender</h1><p>Template not found. Please create templates/index.html</p>",
            status_code=200
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "youtube_available": YOUTUBE_AVAILABLE,
        "spotify_available": spotify_client is not None
    }


@app.get("/api/autocomplete")
async def autocomplete(
    q: str = Query(..., description="Query prefix"),
    model: str = Query("model1", description="Recommendation model (model1 or model2)")
):
    """
    Autocomplete song names from the dataset.
    Returns up to 8 matching songs.
    """
    query = q.lower().strip()
    if not query:
        return []

    # --- Mode 2: Audio Only ---
    if model == "model2":
        if audio_bridge is None or getattr(audio_bridge, 'mappings', None) is None:
            return []
            
        unique_songs = []
        seen = set()
        
        for meta in audio_bridge.mappings:
            song_name = str(meta.get("song", "")).strip()
            if song_name.lower().startswith(query):
                artist_name = clean_artist_name(meta.get("artist", ""))
                key = f"{song_name}::{artist_name}".lower()
                
                if key not in seen:
                    seen.add(key)
                    unique_songs.append({
                        "song": song_name,
                        "artist": artist_name
                    })
                    
                    if len(unique_songs) >= 8:
                        break
        return unique_songs

    # --- Mode 1: Hybrid ---
    if engine is None or engine.song_data is None:
        return []
        
    song_col, artist_col = engine._get_column_names()
    
    # Prefix match
    mask = engine.song_data[song_col].str.lower().str.startswith(query, na=False)
    matches = engine.song_data[mask]
    
    unique_songs = []
    seen = set()
    
    for _, row in matches.head(30).iterrows():
        song_name = str(row.get(song_col, ''))
        artist_name = str(row.get(artist_col, ''))
        
        # Deduplicate
        key = f"{song_name}::{artist_name}".lower()
        if key not in seen:
            seen.add(key)
            unique_songs.append({
                "song": song_name,
                "artist": artist_name
            })
            
            if len(unique_songs) >= 8:
                break
                
    return unique_songs


# =============================================================================
# Song Search — find songs by name (Step 1 of Search-First flow)
# =============================================================================

# Keywords that indicate a non-original version. Used to push remixes/covers
# below originals in the search result ranking.
_REMIX_KEYWORDS: list[str] = [
    "remix", "remixed", "edit", "version", "ver.", "mix",
    "cover", "covered", "instrumental", "acoustic",
    "live", "live at", "live from", "remaster", "remastered",
    "extended", "radio edit", "club mix", "feat.", "ft.",
    "mashup", "bootleg", "dub", "vip", "flip",
    "(reprise)", "reprise", "karaoke", "tribute", "demo",
]


def _is_remix(song_name: str) -> bool:
    """Return True if song_name looks like a remix, cover, or non-original."""
    name_lower = song_name.lower()
    return any(kw in name_lower for kw in _REMIX_KEYWORDS)


class SongSearchItem(BaseModel):
    """One item in a song-search response."""
    song: str
    artist: str
    genre: str
    emotion: str
    is_remix: bool = False


class SongSearchResponse(BaseModel):
    """Response model for /api/song-search."""
    query: str
    results: List[SongSearchItem]
    total: int
    originals_count: int
    remixes_count: int


@app.get("/api/song-search", response_model=SongSearchResponse)
async def song_search(
    q: str = Query(..., description="Song name query (prefix or contains)"),
    artist: Optional[str] = Query(None, description="Optional artist name filter"),
    model: str = Query("model1", description="Dataset source: model1 or model2"),
):
    """
    Search songs by name in the dataset.

    Returns ALL matching songs (not similar-songs), sorted:
      1. Originals first (alphabetically within group)
      2. Remixes / covers / live versions second (alphabetically within group)

    Matching strategy:
      - Prefix matches (song starts with query) come before contains matches
      - Artist filter (optional) applied after matching
      - Pagination is handled client-side

    Used as Step 1 of the Search-First recommendation flow.
    """
    query = q.strip().lower()
    if not query:
        return SongSearchResponse(
            query=q, results=[], total=0, originals_count=0, remixes_count=0
        )

    raw_results: list[dict] = []

    if model == "model2":
        # ── Model 2: Audio-Only dataset ───────────────────────────────────────
        if audio_bridge is None or getattr(audio_bridge, "mappings", None) is None:
            raise HTTPException(status_code=503, detail="Model 2 (Audio-Only) not available")

        seen: set[str] = set()
        prefix_matches: list[dict] = []
        contains_matches: list[dict] = []

        for meta in audio_bridge.mappings:
            song_name = str(meta.get("song", "")).strip()
            artist_name = clean_artist_name(meta.get("artist", ""))
            key = f"{song_name.lower()}::{artist_name.lower()}"
            if key in seen:
                continue

            name_lower = song_name.lower()
            if name_lower.startswith(query):
                seen.add(key)
                prefix_matches.append({"song": song_name, "artist": artist_name,
                                        "genre": str(meta.get("genre", "")),
                                        "emotion": str(meta.get("emotion", ""))})
            elif query in name_lower:
                seen.add(key)
                contains_matches.append({"song": song_name, "artist": artist_name,
                                          "genre": str(meta.get("genre", "")),
                                          "emotion": str(meta.get("emotion", ""))})

        raw_results = prefix_matches + contains_matches

    else:
        # ── Model 1: Hybrid dataset ───────────────────────────────────────────
        if engine is None or engine.song_data is None:
            raise HTTPException(status_code=503, detail="Recommendation engine not initialized")

        song_col, artist_col = engine._get_column_names()
        genre_col = "genre" if "genre" in engine.song_data.columns else None
        emotion_col = "emotion" if "emotion" in engine.song_data.columns else None

        df = engine.song_data
        song_lower = df[song_col].str.lower().fillna("")

        prefix_df = df[song_lower.str.startswith(query, na=False)]
        contains_df = df[song_lower.str.contains(query, regex=False, na=False)]
        # contains_df already includes prefix matches; exclude them to avoid dups
        contains_only_df = contains_df[~contains_df.index.isin(prefix_df.index)]

        seen_keys: set[str] = set()

        def _row_to_dict(row: "pd.Series") -> Optional[dict]:  # type: ignore[name-defined]
            sn = str(row.get(song_col, "")).strip()
            an = clean_artist_name(str(row.get(artist_col, "")))
            key = f"{sn.lower()}::{an.lower()}"
            if key in seen_keys:
                return None
            seen_keys.add(key)
            return {
                "song": sn,
                "artist": an,
                "genre": str(row.get(genre_col, "")) if genre_col else "",
                "emotion": str(row.get(emotion_col, "")) if emotion_col else "",
            }

        prefix_results: list[dict] = []
        for _, row in prefix_df.iterrows():
            d = _row_to_dict(row)
            if d:
                prefix_results.append(d)

        contains_results: list[dict] = []
        for _, row in contains_only_df.iterrows():
            d = _row_to_dict(row)
            if d:
                contains_results.append(d)

        raw_results = prefix_results + contains_results

    # ── Artist filter (optional) ──────────────────────────────────────────────
    if artist:
        artist_lower = artist.strip().lower()
        raw_results = [r for r in raw_results if artist_lower in r["artist"].lower()]

    # ── Attach is_remix flag ──────────────────────────────────────────────────
    for r in raw_results:
        r["is_remix"] = _is_remix(r["song"])

    # ── Sort: originals first, remixes second; alphabetical within each group ─
    originals = sorted([r for r in raw_results if not r["is_remix"]], key=lambda x: x["song"].lower())
    remixes   = sorted([r for r in raw_results if r["is_remix"]],     key=lambda x: x["song"].lower())
    sorted_results = originals + remixes

    items = [SongSearchItem(**r) for r in sorted_results]

    return SongSearchResponse(
        query=q,
        results=items,
        total=len(items),
        originals_count=len(originals),
        remixes_count=len(remixes),
    )


@app.get("/api/search", response_model=SearchResponse)
async def search_songs(
    q: str = Query(..., description="Song name to search for"),
    artist: Optional[str] = Query(None, description="Artist name (optional)"),
    model: str = Query("model1", description="Recommendation model (model1 or model2)"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
):
    """
    Search for similar songs based on song name.
    Returns top 10 most similar songs from the database.
    If user is logged in, automatically saves to search history.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")

    if model == "model2":
        if audio_bridge is None:
            raise HTTPException(status_code=503, detail="Model 2 (Audio-Only) not available")
        m2_res = await asyncio.to_thread(audio_bridge.recommend_from_song, q, artist, 10)
        results = m2_res["recommendations"] if m2_res else []
    else:
        results = await asyncio.to_thread(engine.get_similar_songs, q, artist, 10)

    if not results:
        return SearchResponse(query=q, results=[], count=0)

    songs = [SongResult(**r) for r in results]

    # Auto-save to history if user is logged in
    if DB_AVAILABLE and credentials is not None:
        try:
            from auth import decode_token
            from database import SessionLocal
            payload = decode_token(credentials.credentials)
            if payload is not None:
                db_session = SessionLocal()
                try:
                    user = db_session.query(User).filter(User.id == int(payload.sub)).first()
                    if user:
                        entry = SearchHistory(
                            user_id=user.id,
                            query_song=q,
                            query_artist=artist or "",
                            model_used=model,
                            results_count=len(songs),
                        )
                        db_session.add(entry)
                        db_session.commit()
                finally:
                    db_session.close()
        except Exception as hist_err:
            print(f"[WARNING] Failed to save search history: {hist_err}")

    return SearchResponse(query=q, results=songs, count=len(songs))


@app.get("/api/mood/{mood}", response_model=List[SongResult])
async def get_by_mood(
    mood: str,
    limit: int = Query(10, ge=1, le=50, description="Number of results")
):
    """
    Get song recommendations by mood.
    Available moods: Happy, Sad, Energetic, Calm, Angry
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    valid_moods = ["happy", "sad", "energetic", "calm", "angry"]
    if mood.lower() not in valid_moods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mood. Available: {', '.join(valid_moods)}"
        )
    
    results = engine.get_recommendations_by_mood(mood, top_k=limit)
    return [SongResult(**r) for r in results] if results else []


@app.get("/api/context/{context}", response_model=List[SongResult])
async def get_by_context(
    context: str,
    limit: int = Query(10, ge=1, le=50, description="Number of results")
):
    """
    Get song recommendations by context.
    Available contexts: Party, Workout, Study, Relax, Driving
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    valid_contexts = ["party", "workout", "study", "relax", "driving"]
    if context.lower() not in valid_contexts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid context. Available: {', '.join(valid_contexts)}"
        )
    
    results = engine.get_recommendations_by_context(context, top_k=limit)
    return [SongResult(**r) for r in results] if results else []



# =============================================================================
# Spotify Enrich Endpoint — Disk-Persisted Cache
# =============================================================================
# Cache sống qua server restart — ghi vào file JSON, load lại khi khởi động.
# Chỉ persist kết quả found=True (24h TTL).
# Kết quả thất bại/rate-limit chỉ cache in-memory 60s để retry sớm.
# =============================================================================

import json
import threading as _threading_cache

_SPOTIFY_CACHE_TTL      = 86400   # 24h  — found=True, ghi disk
_SPOTIFY_CACHE_TTL_FAIL = 300     # 5min — found=False/timeout, chỉ in-memory (tránh spam khi rate-limit)


class SpotifyDiskCache:
    """
    Spotify metadata cache với persistence qua JSON file.

    - Load từ disk khi khởi động → không mất cache sau restart.
    - Ghi disk chỉ khi found=True để tránh lưu lỗi 429/timeout.
    - Failed lookups vẫn được cache in-memory 60s để tránh spam API.
    - Thread-safe với threading.Lock (safe ở module-load time, flush là sync I/O).
    """

    CACHE_FILE = os.path.join(os.path.dirname(__file__), "spotify_cache.json")

    def __init__(self):
        self._mem: dict = {}          # key -> {data, expires_at}
        self._lock = _threading_cache.Lock()
        self._dirty = False           # có entry mới chưa flush
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def _load_from_disk(self):
        """Load cache from JSON file, skipping expired entries."""
        if not os.path.exists(self.CACHE_FILE):
            print("[SpotifyCache] No cache file found — starting fresh.")
            return
        try:
            with open(self.CACHE_FILE, "r", encoding="utf-8") as f:
                raw: dict = json.load(f)
            now = time.time()
            loaded = expired = 0
            for key, entry in raw.items():
                if entry.get("expires_at", 0) > now:
                    self._mem[key] = entry
                    loaded += 1
                else:
                    expired += 1
            print(f"[SpotifyCache] Loaded {loaded} entries from disk "
                  f"({expired} expired entries discarded).")
        except Exception as e:
            print(f"[SpotifyCache] Failed to read cache file: {e}")

    def _flush_to_disk(self):
        """Write successful entries (found=True) to JSON file, thread-safe."""
        with self._lock:
            try:
                to_save = {
                    k: v for k, v in self._mem.items()
                    if v.get("data", {}).get("found") is True
                }
                with open(self.CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(to_save, f, ensure_ascii=False, separators=(",", ":"))
                self._dirty = False
            except Exception as e:
                print(f"[SpotifyCache] Failed to write cache: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[dict]:
        """Return cached data if still valid, else None."""
        entry = self._mem.get(key)
        if entry and entry["expires_at"] > time.time():
            return entry["data"]
        return None

    def set(self, key: str, data: dict, ttl: float):
        """Store in memory. If found=True, also flush to disk."""
        self._mem[key] = {"data": data, "expires_at": time.time() + ttl}
        if data.get("found") is True:
            self._dirty = True
            self._flush_to_disk()   # sync write — small file (~KB), fast

    def stats(self) -> dict:
        now = time.time()
        total = len(self._mem)
        alive = sum(1 for v in self._mem.values() if v["expires_at"] > now)
        return {"total_keys": total, "alive_keys": alive,
                "cache_file": self.CACHE_FILE}


# Singleton — initialized once at module load
_spotify_cache = SpotifyDiskCache()


@app.get("/api/spotify-enrich")
async def spotify_enrich(
    q: str = Query(..., description="Song name"),
    artist: Optional[str] = Query(None, description="Artist name (optional)")
):
    """
    Get Spotify metadata (album art, Spotify URL, popularity) for a given song.

    Used by the frontend to enrich song cards with album art after Model 1
    returns recommendations. Makes one Spotify Search API call per song.

    Cache: found=True lưu disk 24h (sống qua restart).
           found=False chỉ in-memory 60s để retry nhanh sau rate-limit.

    Returns:
        { found: bool, album_art, spotify_url, popularity, artist, album }
        Luôn có spotify_search_url làm fallback dù found=False.
    """
    # Fallback URL — Spotify search page, luôn có
    _search_q = f"{q} {artist or ''}".strip().replace(" ", "%20")
    _spotify_search_url = f"https://open.spotify.com/search/{_search_q}"

    if spotify_client is None:
        return {"found": False, "reason": "Spotify client not available",
                "spotify_search_url": _spotify_search_url}

    # ── Cache lookup ──────────────────────────────────────────────────
    cache_key = f"{q.lower()}|{(artist or '').lower()}"
    cached = _spotify_cache.get(cache_key)
    if cached is not None:
        return cached

    # ── Gọi Spotify API ───────────────────────────────────────────────
    # NOTE: timeout=5.0 < Spotify retry-after(5s) để fail fast thay vì block.
    # Khi 429, spotify_client._get() sẽ sleep 5s → vượt timeout → TimeoutError.
    # Đây là intentional: cache 5 min để không spam API.
    try:
        track = await asyncio.wait_for(
            asyncio.to_thread(spotify_client.search_track, q, artist),
            timeout=8.0
        )

        if not track:
            result = {"found": False, "reason": "Song not found on Spotify",
                      "spotify_search_url": _spotify_search_url}
            _spotify_cache.set(cache_key, result, _SPOTIFY_CACHE_TTL_FAIL)
            return result

        result = {
            "found": True,
            "song":        track["name"],
            "artist":      track["artist"],
            "album":       track["album"],
            "album_art":   track["album_art"],
            "spotify_url": track["spotify_url"],
            "spotify_search_url": _spotify_search_url,
            "popularity":  track["popularity"],
            "preview_url": track.get("preview_url"),
            "duration_ms": track["duration_ms"],
            "explicit":    track["explicit"],
        }
        _spotify_cache.set(cache_key, result, _SPOTIFY_CACHE_TTL)   # flush disk
        return result

    except asyncio.TimeoutError:
        print(f"[WARNING] spotify-enrich timeout (likely 429 rate-limit): {q} / {artist}")
        result = {"found": False, "reason": "Spotify API timeout — rate limited, retry in 5 min",
                  "spotify_search_url": _spotify_search_url}
        _spotify_cache.set(cache_key, result, _SPOTIFY_CACHE_TTL_FAIL)
        return result

    except Exception as e:
        print(f"[WARNING] spotify-enrich error: {e}")
        result = {"found": False, "reason": str(e),
                  "spotify_search_url": _spotify_search_url}
        _spotify_cache.set(cache_key, result, 30)  # cache lỗi 30s
        return result


@app.get("/api/spotify-cache-stats")
async def spotify_cache_stats():
    """Debug endpoint: xem trạng thái Spotify cache."""
    return _spotify_cache.stats()


@app.get("/api/youtube", response_model=YouTubeResult)
async def get_youtube_video(
    song: str = Query(..., description="Song name"),
    artist: str = Query(..., description="Artist name")
):
    """
    Get YouTube embed URL for a song.
    Searches YouTube for the official audio/video.
    Falls back to YouTube search URL if server-side search is unavailable.
    """
    if not YOUTUBE_AVAILABLE:
        # Fallback: return YouTube search URL
        query = f"{song} {artist}".strip()
        search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        return YouTubeResult(
            success=True,
            video_id=None,
            embed_url=search_url,
            title=f"Search: {query}",
            message="yt-dlp not installed. Click to search on YouTube."
        )
    
    search_query = f"{song} {artist} official audio"

    try:
        # yt-dlp search options
        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'quiet': True,
            'default_search': 'ytsearch1',
            'no_warnings': True,
            'socket_timeout': 10,  # 10 second timeout
        }
        
        def do_search():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # search_query is passed directly, yt-dlp handles "ytsearch1:" prefix if default_search is set
                # or we can be explicit: f"ytsearch1:{search_query}"
                info = ydl.extract_info(search_query, download=False)
                return info
        
        # Run in thread pool to avoid blocking
        result = await asyncio.to_thread(do_search)
        
        if 'entries' in result and result['entries']:
            video = result['entries'][0]
            video_id = video.get('id')
            video_title = video.get('title', f"{song} - {artist}")
            thumbnail = video.get('thumbnail', '')
            
            return YouTubeResult(
                success=True,
                video_id=video_id,
                embed_url=f"https://www.youtube.com/embed/{video_id}",
                title=video_title,
                thumbnail=thumbnail
            )
            
        # No results found, fallback to search URL
        query = f"{song} {artist}".strip()
        search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        return YouTubeResult(
            success=True,
            video_id=None,
            embed_url=search_url,
            title=f"Search: {query}",
            message="No exact match found. Click to search on YouTube."
        )
        
    except Exception as e:
        # Graceful fallback on any error (network, DNS, timeout, etc.)
        print(f"[INFO] YouTube API unavailable, using search fallback: {e}")
        query = f"{song} {artist}".strip()
        search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        
        return YouTubeResult(
            success=True,
            video_id=None,
            embed_url=search_url,
            title=f"Search: {query}",
            message="Server-side YouTube search unavailable. Click to search on YouTube."
        )


# =============================================================================
# Image-Based Recommendation Endpoint
# =============================================================================

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


@app.post("/api/recommend-from-image", response_model=ImageRecommendationResponse)
async def recommend_from_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, GIF)")
):
    """
    Get music recommendations based on an uploaded image.
    
    The system uses CLIP (Contrastive Language-Image Pre-training) to analyze
    the image and detect:
    - Mood: Happy, Sad, Energetic, Calm, Angry
    - Context: Party, Workout, Study, Relax, Driving
    
    Then returns appropriate music recommendations.
    
    **Supported formats**: JPEG, PNG, WebP, GIF
    **Max file size**: 10MB
    """
    # Check if image processor is available
    if not IMAGE_PROCESSOR_AVAILABLE or image_classifier is None:
        return ImageRecommendationResponse(
            success=False,
            message="Image-based recommendation is not available. CLIP model not installed."
        )
    
    if engine is None:
        return ImageRecommendationResponse(
            success=False,
            message="Recommendation engine not initialized."
        )
    
    # Validate file type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_TYPES:
        return ImageRecommendationResponse(
            success=False,
            message=f"Invalid file type: {content_type}. Allowed: JPEG, PNG, WebP, GIF"
        )
    
    # Read file content
    try:
        image_bytes = await file.read()
        
        # Check file size
        if len(image_bytes) > MAX_IMAGE_SIZE:
            return ImageRecommendationResponse(
                success=False,
                message=f"File too large. Maximum size: 10MB"
            )
        
        # Analyze image with CLIP
        label, label_type, confidence = await asyncio.to_thread(
            image_classifier.analyze_image, image_bytes
        )
        
        # Get recommendations based on detected mood or context
        if label_type == "mood":
            results = engine.get_recommendations_by_mood(label, top_k=10)
        else:  # context
            results = engine.get_recommendations_by_context(label, top_k=10)
        
        if not results:
            return ImageRecommendationResponse(
                success=True,
                detected_label=label,
                detected_type=label_type,
                confidence=confidence,
                recommendations=[],
                message=f"Detected {label} ({label_type}), but no matching songs found."
            )
        
        return ImageRecommendationResponse(
            success=True,
            detected_label=label,
            detected_type=label_type,
            confidence=confidence,
            recommendations=[SongResult(**r) for r in results],
            message=f"Detected: {label} ({label_type}) with {confidence:.0%} confidence"
        )
        
    except ValueError as e:
        return ImageRecommendationResponse(
            success=False,
            message=f"Could not process image: {str(e)}"
        )
    except Exception as e:
        print(f"[ERROR] Image analysis failed: {e}")
        return ImageRecommendationResponse(
            success=False,
            message="An error occurred while analyzing the image. Please try again."
        )


# =============================================================================
# Image-Based Recommendation v2 Endpoint (Model 2 - Audio-Only + CLIP Bridge)
# =============================================================================

@app.post("/api/recommend/image-v2", response_model=ImageRecommendationResponse)
async def recommend_from_image_v2(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, GIF)")
):
    """
    Get music recommendations using Model 2 (Audio-Only) based on an uploaded image.

    Pipeline:
      Image → CLIP (mood/context detection) → Audio Profile →
      Model 2 encode (32D) → FAISS search → top 10 songs

    Compared to /api/recommend-from-image (Model 1 Hybrid), this endpoint
    uses purely audio features without lyrics/text, giving a different
    audio-characteristic-based recommendation.
    """
    # Check if audio bridge is available (requires trained Model 2)
    if audio_bridge is None:
        return ImageRecommendationResponse(
            success=False,
            message="Model 2 (Audio-Only) is not available. The model may not be trained yet."
        )

    if not IMAGE_PROCESSOR_AVAILABLE or image_classifier is None:
        return ImageRecommendationResponse(
            success=False,
            message="CLIP model not available. Image-based recommendation disabled."
        )

    # Validate file type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_TYPES:
        return ImageRecommendationResponse(
            success=False,
            message=f"Invalid file type: {content_type}. Allowed: JPEG, PNG, WebP, GIF"
        )

    try:
        image_bytes = await file.read()

        if len(image_bytes) > MAX_IMAGE_SIZE:
            return ImageRecommendationResponse(
                success=False,
                message="File too large. Maximum size: 10MB"
            )

        # Step 1: Use CLIP to detect mood/context from the image (same as Model 1)
        label, label_type, confidence = await asyncio.to_thread(
            image_classifier.analyze_image, image_bytes
        )

        # Step 2: Pass detected label through Audio Bridge → Model 2 → FAISS
        results = await asyncio.to_thread(
            audio_bridge.recommend_from_label, label, 10
        )

        if not results:
            return ImageRecommendationResponse(
                success=True,
                detected_label=label,
                detected_type=label_type,
                confidence=confidence,
                recommendations=[],
                message=f"Detected {label} ({label_type}), but no matching songs found."
            )

        return ImageRecommendationResponse(
            success=True,
            detected_label=label,
            detected_type=label_type,
            confidence=confidence,
            recommendations=[SongResult(**r) for r in results],
            message=f"[Model 2] Detected: {label} ({label_type}) with {confidence:.0%} confidence"
        )

    except ValueError as e:
        return ImageRecommendationResponse(
            success=False,
            message=f"Could not process image: {str(e)}"
        )
    except Exception as e:
        print(f"[ERROR] Image v2 analysis failed: {e}")
        print(traceback.format_exc())
        return ImageRecommendationResponse(
            success=False,
            message=f"An error occurred while analyzing the image: {e}"
        )


# =============================================================================
# Auth Endpoints
# =============================================================================

@app.post("/api/auth/register", response_model=AuthResponse)
async def register(
    body: RegisterRequest,
    db: "Session" = Depends(get_db),
):
    """Register a new user account."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")

    # Check uniqueness
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=body.username,
        email=body.email,
        password_hash=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.username)
    return AuthResponse(
        token=token,
        user={"id": user.id, "username": user.username, "email": user.email},
    )


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(
    body: LoginRequest,
    db: "Session" = Depends(get_db),
):
    """Login with username + password, returns JWT token."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")

    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(user.id, user.username)
    return AuthResponse(
        token=token,
        user={"id": user.id, "username": user.username, "email": user.email},
    )


@app.get("/api/auth/me")
async def get_me(current_user: "User" = Depends(get_current_user)):
    """Get current authenticated user info."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    return {"id": current_user.id, "username": current_user.username, "email": current_user.email}


# =============================================================================
# Search History Endpoints
# =============================================================================

@app.get("/api/history", response_model=List[HistoryItem])
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    current_user: "User" = Depends(get_current_user),
    db: "Session" = Depends(get_db),
):
    """Get search history for the authenticated user (newest first)."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    items = (
        db.query(SearchHistory)
        .filter(SearchHistory.user_id == current_user.id)
        .order_by(SearchHistory.searched_at.desc())
        .limit(limit)
        .all()
    )
    return items


@app.delete("/api/history/{item_id}")
async def delete_history_item(
    item_id: int,
    current_user: "User" = Depends(get_current_user),
    db: "Session" = Depends(get_db),
):
    """Delete a single search history entry."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    item = db.query(SearchHistory).filter(
        SearchHistory.id == item_id,
        SearchHistory.user_id == current_user.id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="History entry not found")
    db.delete(item)
    db.commit()
    return {"success": True}


@app.delete("/api/history")
async def clear_history(
    current_user: "User" = Depends(get_current_user),
    db: "Session" = Depends(get_db),
):
    """Clear all search history for the authenticated user."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    db.query(SearchHistory).filter(SearchHistory.user_id == current_user.id).delete()
    db.commit()
    return {"success": True}


# =============================================================================
# Playlist Endpoints
# =============================================================================

@app.post("/api/playlists", response_model=PlaylistOut)
async def create_playlist(
    body: PlaylistCreateRequest,
    current_user: "User" = Depends(get_current_user),
    db: "Session" = Depends(get_db),
):
    """Create a new playlist for the authenticated user."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    playlist = Playlist(
        user_id=current_user.id,
        name=body.name,
        description=body.description,
    )
    db.add(playlist)
    db.commit()
    db.refresh(playlist)
    result = PlaylistOut(
        id=playlist.id,
        name=playlist.name,
        description=playlist.description,
        created_at=playlist.created_at,
        updated_at=playlist.updated_at,
        track_count=0,
    )
    return result


@app.get("/api/playlists", response_model=List[PlaylistOut])
async def list_playlists(
    current_user: "User" = Depends(get_current_user),
    db: "Session" = Depends(get_db),
):
    """List all playlists for the authenticated user."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    playlists = (
        db.query(Playlist)
        .filter(Playlist.user_id == current_user.id)
        .order_by(Playlist.updated_at.desc())
        .all()
    )
    return [
        PlaylistOut(
            id=p.id,
            name=p.name,
            description=p.description,
            created_at=p.created_at,
            updated_at=p.updated_at,
            track_count=len(p.tracks),
        )
        for p in playlists
    ]


@app.get("/api/playlists/{playlist_id}", response_model=PlaylistDetailOut)
async def get_playlist(
    playlist_id: int,
    current_user: "User" = Depends(get_current_user),
    db: "Session" = Depends(get_db),
):
    """Get a playlist with all its tracks."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    playlist = db.query(Playlist).filter(
        Playlist.id == playlist_id,
        Playlist.user_id == current_user.id,
    ).first()
    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")
    return PlaylistDetailOut(
        id=playlist.id,
        name=playlist.name,
        description=playlist.description,
        created_at=playlist.created_at,
        updated_at=playlist.updated_at,
        tracks=[TrackOut.model_validate(t) for t in playlist.tracks],
    )


@app.put("/api/playlists/{playlist_id}", response_model=PlaylistOut)
async def update_playlist(
    playlist_id: int,
    body: PlaylistCreateRequest,
    current_user: "User" = Depends(get_current_user),
    db: "Session" = Depends(get_db),
):
    """Update playlist name and/or description."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    playlist = db.query(Playlist).filter(
        Playlist.id == playlist_id,
        Playlist.user_id == current_user.id,
    ).first()
    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")
    playlist.name = body.name
    playlist.description = body.description
    playlist.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(playlist)
    return PlaylistOut(
        id=playlist.id,
        name=playlist.name,
        description=playlist.description,
        created_at=playlist.created_at,
        updated_at=playlist.updated_at,
        track_count=len(playlist.tracks),
    )


@app.delete("/api/playlists/{playlist_id}")
async def delete_playlist(
    playlist_id: int,
    current_user: "User" = Depends(get_current_user),
    db: "Session" = Depends(get_db),
):
    """Delete a playlist (and all its tracks via cascade)."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    playlist = db.query(Playlist).filter(
        Playlist.id == playlist_id,
        Playlist.user_id == current_user.id,
    ).first()
    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")
    db.delete(playlist)
    db.commit()
    return {"success": True}


@app.post("/api/playlists/{playlist_id}/tracks", response_model=TrackOut)
async def add_track_to_playlist(
    playlist_id: int,
    body: TrackAddRequest,
    current_user: "User" = Depends(get_current_user),
    db: "Session" = Depends(get_db),
):
    """Add a track to a playlist."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    playlist = db.query(Playlist).filter(
        Playlist.id == playlist_id,
        Playlist.user_id == current_user.id,
    ).first()
    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")

    # Assign position = last + 1
    max_pos = max((t.position for t in playlist.tracks), default=-1)
    track = PlaylistTrack(
        playlist_id=playlist_id,
        song_name=body.song_name,
        artist_name=body.artist_name,
        genre=body.genre,
        emotion=body.emotion,
        spotify_url=body.spotify_url,
        album_art_url=body.album_art_url,
        position=max_pos + 1,
    )
    db.add(track)
    playlist.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(track)
    return TrackOut.model_validate(track)


@app.delete("/api/playlists/{playlist_id}/tracks/{track_id}")
async def remove_track_from_playlist(
    playlist_id: int,
    track_id: int,
    current_user: "User" = Depends(get_current_user),
    db: "Session" = Depends(get_db),
):
    """Remove a track from a playlist."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    playlist = db.query(Playlist).filter(
        Playlist.id == playlist_id,
        Playlist.user_id == current_user.id,
    ).first()
    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")
    track = db.query(PlaylistTrack).filter(
        PlaylistTrack.id == track_id,
        PlaylistTrack.playlist_id == playlist_id,
    ).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    db.delete(track)
    playlist.updated_at = datetime.utcnow()
    db.commit()
    return {"success": True}


# =============================================================================
# Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
