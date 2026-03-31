"""
Music Recommender Web API - FastAPI Backend Application.

================================================================================
PURPOSE:
================================================================================
This module serves as the web interface for the Music Recommendation System.
It exposes the recommendation engine's functionality through RESTful API endpoints,
allowing users to interact with the system via a modern web browser.

================================================================================
ARCHITECTURE:
================================================================================
    Frontend (HTML/CSS/JS)
           │
           │ HTTP/REST
           ▼
    FastAPI Application (this file)
           │
           │ Python calls
           ▼
    MusicRecommendationEngine
           │
     ┌─────┴─────┐
     ▼           ▼
  PyTorch    FAISS Index
   Model     (551K songs)

================================================================================
API ENDPOINTS:
================================================================================
- GET /              : Serve main HTML page
- GET /api/health    : Health check endpoint
- GET /api/search    : Search for similar songs by name
- GET /api/mood/{mood}      : Get recommendations by mood
- GET /api/context/{context}: Get recommendations by activity context
- GET /api/youtube   : Get YouTube embed URL for a song

================================================================================
FILE STRUCTURE:
================================================================================
- Pydantic Models: SongResult, YouTubeResult, SearchResponse
- lifespan(): Context manager for startup/shutdown (loads model)
- index(): Serves the main HTML page
- search_songs(): Similarity search endpoint
- get_by_mood(): Mood-based recommendations
- get_by_context(): Context-based recommendations
- get_youtube_video(): YouTube integration

================================================================================
RELATED FILES:
================================================================================
- templates/index.html: Frontend HTML template
- static/css/style.css: Dark theme CSS styling
- static/js/main.js: Frontend JavaScript logic
- hybrid_music_engine/inference.py: MusicRecommendationEngine class

================================================================================
USAGE:
================================================================================
    cd web_app
    python -m uvicorn app:app --host 127.0.0.1 --port 8000

Then open: http://127.0.0.1:8000
API Docs: http://127.0.0.1:8000/docs

================================================================================
Author: Graduation Project
Created: 2026-01-19
================================================================================
"""

from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import sys
import os

# Add parent directory to path to import hybrid_music_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import download helper for Vercel deployment
try:
    from download_helper import ensure_files_ready
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
    IMAGE_PROCESSOR_AVAILABLE = False
    print(f"[WARNING] ImageMoodClassifier import failed: {e}. Image-based recommendation disabled.")

# Try to import SpotifyClient for enriched search and album art
try:
    from spotify_client import SpotifyClient
    SPOTIFY_CLIENT_AVAILABLE = True
except ImportError as e:
    SPOTIFY_CLIENT_AVAILABLE = False
    print(f"[WARNING] SpotifyClient import failed: {e}. Spotify enrichment disabled.")

# Try to import CLIPAudioBridge for Model 2 image recommendations
try:
    from audio_model.clip_audio_bridge import CLIPAudioBridge
    AUDIO_BRIDGE_AVAILABLE = True
except ImportError as e:
    AUDIO_BRIDGE_AVAILABLE = False
    print(f"[WARNING] CLIPAudioBridge import failed: {e}. Model 2 image recommendation disabled.")


# =============================================================================
# Pydantic Models
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
                artist_name = str(meta.get("artist", "")).strip()
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


@app.get("/api/search", response_model=SearchResponse)
async def search_songs(
    q: str = Query(..., description="Song name to search for"),
    artist: Optional[str] = Query(None, description="Artist name (optional)"),
    model: str = Query("model1", description="Recommendation model (model1 or model2)")
):
    """
    Search for similar songs based on song name.
    Returns top 10 most similar songs from the database.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    if model == "model2":
        if audio_bridge is None:
            raise HTTPException(status_code=503, detail="Model 2 (Audio-Only) not available")
        # Run in thread since recommend_from_song isn't async
        m2_res = await asyncio.to_thread(audio_bridge.recommend_from_song, q, artist, 10)
        results = m2_res["recommendations"] if m2_res else []
    else:
        # Run Model 1 in thread just in case it blocks
        results = await asyncio.to_thread(engine.get_similar_songs, q, artist, 10)
    
    if not results:
        return SearchResponse(query=q, results=[], count=0)
    
    songs = [SongResult(**r) for r in results]
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
# Spotify Enrich Endpoint - returns album art + Spotify metadata for any song
# =============================================================================

@app.get("/api/spotify-enrich")
async def spotify_enrich(
    q: str = Query(..., description="Song name"),
    artist: Optional[str] = Query(None, description="Artist name (optional)")
):
    """
    Get Spotify metadata (album art, Spotify URL, popularity) for a given song.

    Used by the frontend to enrich song cards with album art after Model 1
    returns recommendations. Makes one Spotify Search API call per song.

    Returns:
        { found: bool, album_art, spotify_url, popularity, artist, album }
    """
    if spotify_client is None:
        return {"found": False, "reason": "Spotify client not available"}

    try:
        track = await asyncio.to_thread(spotify_client.search_track, q, artist)
        if not track:
            return {"found": False, "reason": "Song not found on Spotify"}

        return {
            "found": True,
            "song": track["name"],
            "artist": track["artist"],
            "album": track["album"],
            "album_art": track["album_art"],
            "spotify_url": track["spotify_url"],
            "popularity": track["popularity"],
            "preview_url": track.get("preview_url"),
            "duration_ms": track["duration_ms"],
            "explicit": track["explicit"],
        }
    except Exception as e:
        print(f"[WARNING] spotify-enrich error: {e}")
        return {"found": False, "reason": str(e)}


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
        return ImageRecommendationResponse(
            success=False,
            message="An error occurred while analyzing the image. Please try again."
        )


# =============================================================================
# Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
