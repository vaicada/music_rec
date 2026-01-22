# Prompt: Tạo Web Application cho Music Recommendation System

## Mục tiêu

Tạo một trang web với giao diện hiện đại để tìm kiếm và gợi ý nhạc, sử dụng backend Python hiện có từ project Hybrid Music Recommender.

---

## Thông tin về Backend hiện tại

### Cấu trúc project

```
music_recommender/
├── hybrid_music_engine/          # Core recommendation engine
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── inference.py              # MusicRecommendationEngine class
│   ├── model.py                  # HybridMusicModel (PyTorch)
│   ├── processors.py             # Text, Audio, Metadata processors
│   └── trainer.py                # Training utilities
├── models/                       # Trained model files
│   ├── best_model.pth            # PyTorch model weights
│   ├── faiss_index.bin           # FAISS similarity index
│   └── mappings.pkl              # Song ID mappings
├── data/processed/               # Processed data
│   └── train.csv                 # Training data with lyrics, features
└── interactive_demo.py           # Console-based demo (reference)
```

### API chính từ `inference.py`

```python
from hybrid_music_engine import get_config
from hybrid_music_engine.inference import MusicRecommendationEngine

# Khởi tạo engine
config = get_config()
engine = MusicRecommendationEngine(config)

# Load model và index
engine.load_model("models/best_model.pth")
engine.load_index("models/faiss_index.bin")
engine.load_song_data("data/processed/train.csv")

# API Methods:
# 1. Tìm bài hát tương tự
results = engine.get_similar_songs(song_name="Shape of You", artist_name="Ed Sheeran", top_k=10)
# Returns: List[dict] với keys: 'song', 'artist', 'genre', 'emotion', 'similarity'

# 2. Gợi ý theo mood
results = engine.get_recommendations_by_mood(mood="Happy", top_k=10)
# Moods: Happy, Sad, Energetic, Calm, Angry

# 3. Gợi ý theo context
results = engine.get_recommendations_by_context(context="Party", top_k=10)
# Contexts: Party, Workout, Study, Relax, Driving
```

### Cấu trúc dữ liệu trả về

```python
# Mỗi bài hát trả về có format:
{
    'song': 'Shape of You',           # Tên bài hát
    'artist': 'Ed Sheeran',           # Nghệ sĩ
    'genre': 'Pop',                   # Thể loại
    'emotion': 'Happy',               # Cảm xúc
    'similarity': 0.95                # Điểm tương đồng (chỉ có khi dùng get_similar_songs)
}
```

---

## Yêu cầu cho Web Application

### 1. Frontend (HTML/CSS/JavaScript)

- **Trang chủ (index.html):**
  - Search bar ở trung tâm với placeholder "Nhập tên bài hát hoặc nghệ sĩ..."
  - Nút tìm kiếm (icon kính lúp)
  - Filter buttons: "Tất cả", "Theo Mood", "Theo Context"
  - Dropdown cho Mood (Happy, Sad, Energetic, Calm, Angry)
  - Dropdown cho Context (Party, Workout, Study, Relax, Driving)

- **Kết quả tìm kiếm:**
  - Grid hoặc list hiển thị các bài hát
  - Mỗi card bài hát gồm:
    - Tên bài hát (có thể click)
    - Tên nghệ sĩ
    - Thể loại và Emotion (badges)
    - Điểm tương đồng (nếu có)
    - Nút Play YouTube
    - YouTube embed player (có thể ẩn/hiện)

- **Thiết kế:**
  - Modern, dark theme với gradient accent
  - Responsive (mobile-friendly)
  - Smooth animations khi load kết quả
  - Loading spinner khi đang tìm kiếm

### 2. Backend API (FastAPI)

- **Endpoints cần tạo:**

  ```
  GET  /api/search?q=<query>&artist=<optional>    # Tìm bài hát tương tự
  GET  /api/mood/{mood}?limit=10                  # Gợi ý theo mood
  GET  /api/context/{context}?limit=10            # Gợi ý theo context
  GET  /api/youtube?song=<name>&artist=<name>     # Tìm YouTube link
  ```

- **Tích hợp với engine:**
  - Import `MusicRecommendationEngine` từ `hybrid_music_engine`
  - Khởi tạo engine một lần khi server start (lifespan event)
  - Sử dụng async/await cho YouTube API calls
  - Trả về JSON response với Pydantic models
  - Auto-generated Swagger docs tại `/docs`

### 3. Tích hợp YouTube

- Sử dụng YouTube Data API v3 hoặc youtube-search-python
- Tự động tìm video dựa trên "[Tên bài hát] - [Nghệ sĩ] official"
- Nhúng video bằng iframe YouTube embed
- Format embed URL: `https://www.youtube.com/embed/{video_id}`

---

## Cấu trúc file đề xuất

```plaintext
music_recommender/
├── web_app/
│   ├── app.py                    # FastAPI backend
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css         # Styling
│   │   └── js/
│   │       └── main.js           # Frontend logic
│   └── templates/
│       └── index.html            # Main page
└── requirements.txt              # Thêm: fastapi, uvicorn, youtube-search-python
```

---

## Ví dụ Code Snippet

### Backend (app.py)

```python
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import sys
sys.path.insert(0, '..')
from hybrid_music_engine import get_config
from hybrid_music_engine.inference import MusicRecommendationEngine
from youtube_search import YoutubeSearch

# Pydantic models
class SongResult(BaseModel):
    song: str
    artist: str
    genre: str
    emotion: str
    similarity: Optional[float] = None

class YouTubeResult(BaseModel):
    video_id: str
    embed_url: str

# Global engine
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global engine
    config = get_config()
    engine = MusicRecommendationEngine(config)
    engine.load_model("../models/best_model.pth")
    engine.load_index()
    engine.load_song_data("../data/processed/train.csv")
    yield
    # Shutdown: cleanup if needed

app = FastAPI(title="Music Recommender API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/search", response_model=List[SongResult])
async def search(q: str = Query(...), artist: Optional[str] = None):
    results = engine.get_similar_songs(q, artist, top_k=10)
    return results

@app.get("/api/mood/{mood}", response_model=List[SongResult])
async def get_by_mood(mood: str, limit: int = 10):
    return engine.get_recommendations_by_mood(mood, top_k=limit)

@app.get("/api/context/{context}", response_model=List[SongResult])
async def get_by_context(context: str, limit: int = 10):
    return engine.get_recommendations_by_context(context, top_k=limit)

@app.get("/api/youtube", response_model=YouTubeResult)
async def get_youtube(song: str, artist: str):
    search_query = f"{song} {artist} official audio"
    # Run synchronous YoutubeSearch in a separate thread to avoid blocking the event loop
    results = await asyncio.to_thread(YoutubeSearch, search_query, max_results=1)
    results = results.to_dict()
    if results:
        video_id = results[0]['id']
        return {"video_id": video_id, "embed_url": f"https://www.youtube.com/embed/{video_id}"}
    raise HTTPException(status_code=404, detail="Video not found")
```

### Frontend JavaScript (main.js)

```javascript
async function searchSongs() {
    const query = document.getElementById('searchInput').value;
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const songs = await response.json();
    displayResults(songs);
}

async function playOnYouTube(song, artist) {
    const response = await fetch(`/api/youtube?song=${encodeURIComponent(song)}&artist=${encodeURIComponent(artist)}`);
    const data = await response.json();
    if (data.embed_url) {
        document.getElementById('youtube-player').innerHTML = 
            `<iframe width="100%" height="315" src="${data.embed_url}" frameborder="0" allowfullscreen></iframe>`;
    }
}
```

---

## Lưu ý quan trọng

1. **Engine initialization** tốn thời gian (~30s-1 phút lần đầu load BERT model)
2. Path đến model files cần điều chỉnh tùy vào vị trí chạy server
3. Cần xử lý trường hợp không tìm thấy bài hát trong database
4. YouTube API có rate limit, nên cache kết quả nếu cần
5. Sử dụng FastAPI với uvicorn để có async support và hiệu năng cao

---

## Chạy thử

```bash
cd web_app
pip install fastapi uvicorn youtube-search-python
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# Mở http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

*Prompt này được tạo dựa trên cấu trúc project Music Recommender hiện tại.*
