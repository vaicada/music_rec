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

from fastapi import FastAPI, Query, HTTPException
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
    print("⚠️  Warning: download_helper.py not found. Auto-download disabled.")
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


# =============================================================================
# Global Engine Instance
# =============================================================================

engine: Optional[MusicRecommendationEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads the recommendation engine on startup.
    """
    global engine
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
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize engine: {e}")
        print("[WARNING] API will start but recommendations won't work.")
        engine = None
    
    yield
    
    # Cleanup on shutdown
    print("\nShutting down Music Recommender API...")
    engine = None


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
        "youtube_available": YOUTUBE_AVAILABLE
    }


@app.get("/api/search", response_model=SearchResponse)
async def search_songs(
    q: str = Query(..., description="Song name to search for"),
    artist: Optional[str] = Query(None, description="Artist name (optional)")
):
    """
    Search for similar songs based on song name.
    Returns top 10 most similar songs from the database.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    results = engine.get_similar_songs(q, artist, top_k=10)
    
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
# Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
