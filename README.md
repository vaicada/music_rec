---
title: Hybrid Music Recommender
emoji: 🎶
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

<div align="center">
  <h1>🎶 Hybrid Music Recommendation Engine</h1>
  <p>A cutting-edge, dual-model AI system that recommends music based on lyrics, audio features, visual moods, and context.</p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python" />
    <img src="https://img.shields.io/badge/FastAPI-0.104-green?style=flat-square&logo=fastapi" />
    <img src="https://img.shields.io/badge/PyTorch-2.1-orange?style=flat-square&logo=pytorch" />
    <img src="https://img.shields.io/badge/FAISS-1.7.4-purple?style=flat-square" />
    <img src="https://img.shields.io/badge/Deploy-HuggingFace%20Spaces-yellow?style=flat-square&logo=huggingface" />
  </p>
</div>

<p align="center">
  <img src="visualizations/screenshot_search.png" width="90%" alt="Music Recommender Search Interface" />
  <br><em>Giao diện tìm kiếm trực quan trả về các bài hát tương đồng kèm độ tin cậy</em>
</p>

---

## 📌 Tổng Quan Dự Án

Finding the perfect song isn't just about matching genres — it's about matching **emotions, situations, and the acoustic DNA** of the listener's preferences.

Đây là một web application hiệu năng cao được cung cấp bởi **Machine Learning**, vượt ra ngoài collaborative filtering thông thường. Bằng cách phân tích **1.7 triệu bài hát**, hệ thống hiểu được mối quan hệ tinh tế giữa lời bài hát, cấu hình âm thanh (tempo, danceability, energy) và trạng thái tâm trí của người nghe.

---

## ✨ Tính Năng Chính

| Tính năng | Mô tả |
|---|---|
| 🔍 **Smart Search & Autocomplete** | Gõ tên bài hát → gợi ý tức thì từ dataset 441K bài |
| 🎭 **Vibe & Context Matching** | Học tập, Workout, Buồn bã → hệ thống map mood sang vector tức thì |
| 🖼️ **Image-to-Music (Multimodal)** | Upload ảnh → CLIP model nhận diện mood → gợi ý nhạc phù hợp |
| 🤖 **Dual AI Engine** | Chuyển đổi liền mạch giữa 2 mô hình AI khác nhau |
| ▶️ **In-App YouTube Playback** | Nghe nhạc trực tiếp không cần rời app |
| 🎵 **Spotify Integration** | Album art, metadata, và link Spotify cho mỗi bài hát |
| 👤 **User Auth & History** | Đăng ký/đăng nhập, lưu lịch sử tìm kiếm và playlist |

<p align="center">
  <img src="visualizations/screenshot_image_mood.png" width="80%" alt="Image to Mood Interface" style="border-radius: 8px; margin-top: 10px;" />
  <br><em>Upload một tấm ảnh bãi biển → AI tự động nhận diện mood "Energetic" với độ tự tin 91%!</em>
</p>

---

## 🧠 Kiến Trúc AI: Dual Models

Hệ thống chạy **hai mô hình AI riêng biệt** để phục vụ các nhu cầu khác nhau:

### Model 1: Hybrid Content-Based (Supervised)

- **Dataset:** 490,000+ bài hát với nhãn cảm xúc được phân loại (Joy, Sadness, Anger, Love, Fear, ...)
- **Cách hoạt động:** Kết hợp **NLP (BERT)** trên lời bài hát với các đặc trưng âm thanh được chuẩn hóa
- **FAISS Index:** `faiss_index.bin` — 135MB, 441K vectors
- **Tốt nhất khi:** Cần bài hát mang ý nghĩa ngữ nghĩa và cảm xúc cụ thể

### Model 2: Audio Autoencoder (Unsupervised)

- **Dataset:** 1.2 triệu bài hát toàn cầu trải qua 7 thập kỷ
- **Cách hoạt động:** Không dùng nhãn thủ công. Sử dụng **Audio Autoencoder** để nén bài hát thành không gian ẩn 8 chiều (latent space)
- **FAISS Index:** `tracks_faiss.index` — 133MB, 1.2M vectors (8D latent)
- **Tốt nhất khi:** Khám phá nhạc đa dạng dựa thuần túy vào tương đồng âm thanh

```
Input Query (text/audio/image)
        │
        ▼
┌──────────────────────────────────────────┐
│              Web App (FastAPI)            │
│  ┌────────────────┐  ┌─────────────────┐ │
│  │   Model 1      │  │    Model 2      │ │
│  │ Hybrid BERT +  │  │ Audio Autoenc.  │ │
│  │ Audio Features │  │ + CLIP Bridge   │ │
│  └───────┬────────┘  └────────┬────────┘ │
│          │    FAISS Search    │          │
│          ▼                    ▼          │
│     faiss_index.bin    tracks_faiss.index│
└──────────────────────────────────────────┘
        │
        ▼
 Spotify Enrichment → YouTube Playback
```

---

## 📁 Cấu Trúc Dự Án

```
music_recommender/
│
├── 📂 web_app/                     # Backend FastAPI application
│   ├── app.py                      # Main FastAPI app, tất cả API endpoints
│   ├── auth.py                     # JWT authentication (đăng ký/đăng nhập)
│   ├── database.py                 # SQLAlchemy models, PostgreSQL/SQLite
│   ├── download_helper.py          # Tự động tải model files từ Google Drive
│   ├── requirements-deploy.txt     # Dependencies tối thiểu cho deploy
│   └── 📂 static/                  # Frontend assets
│       ├── index.html              # Trang chủ
│       ├── main.js                 # Logic frontend, API calls
│       └── style.css               # CSS styling
│
├── 📂 hybrid_music_engine/         # Model 1: Hybrid Engine
│   ├── __init__.py                 # Package init, expose API chính
│   ├── model.py                    # Kiến trúc mạng nơ-ron PyTorch
│   ├── processors.py               # Xử lý text, audio features, embeddings
│   ├── inference.py                # Inference engine, FAISS search
│   ├── trainer.py                  # Training pipeline
│   ├── config.py                   # Hyperparameters và cấu hình
│   ├── image_processor.py          # CLIP image-to-mood bridge
│   └── logger.py                   # Custom logging
│
├── 📂 audio_model/                 # Model 2: Audio Autoencoder
│   ├── model2.py                   # Autoencoder architecture (8D latent)
│   ├── train_audio_model.py        # Training script
│   ├── build_audio_faiss.py        # Xây dựng FAISS index 1.2M bài hát
│   ├── clip_audio_bridge.py        # CLIP → Audio feature mapping
│   ├── config2.py                  # Model 2 configuration
│   ├── prepare_audio_data.py       # Chuẩn bị dữ liệu audio
│   └── prepare_tracks_data.py      # Chuẩn bị tracks dataset
│
├── 📂 models/                      # Model artifacts (large files - GDrive)
│   ├── best_model.pth              # Model 1 weights (~5MB, trong repo)
│   ├── faiss_index.bin             # Model 1 FAISS index (~135MB)
│   ├── faiss_index.bin.mappings.pkl # Model 1 metadata mappings (~71MB)
│   ├── song_metadata.csv           # Fallback metadata (~29MB, trong repo)
│   ├── autoencoder_model2.pth      # Model 2 weights (~8KB)
│   ├── tracks_faiss.index          # Model 2 FAISS index (8D, ~133MB)
│   ├── tracks_faiss.index.mappings.pkl  # Model 2 mappings (~104MB)
│   └── tracks_stats.json           # Audio normalization statistics
│
├── 📂 data/                        # Datasets (không commit lên Git)
│   └── processed/
│       └── train.csv               # 441K bài hát Model 1 (~964MB)
│
├── 📂 dataset/                     # Raw datasets & notebooks
├── 📂 visualizations/              # Charts, plots, screenshots
├── 📂 tests/                       # Unit & integration tests
│   └── integration/
│       └── test_app.py
│
├── spotify_client.py               # Spotify Web API client
├── prepare_data.py                 # Data preprocessing pipeline
├── train_improved.py               # Model 1 training (full pipeline)
├── build_faiss_index.py            # Xây dựng FAISS index cho Model 1
├── evaluate_final.py               # Evaluation metrics
├── Dockerfile                      # Docker multi-stage build
├── requirements.txt                # Tất cả dependencies
├── .env                            # Biến môi trường (KHÔNG commit!)
└── README.md                       # File này
```

---

## 🛠 Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | FastAPI 0.104, Uvicorn (ASGI), Python 3.11 |
| **Machine Learning** | PyTorch 2.1, FAISS-CPU 1.7.4, Sentence-Transformers (BERT) |
| **Computer Vision** | OpenAI CLIP (image-to-mood) |
| **Data Science** | Pandas 2.1, NumPy 1.26, Scikit-Learn 1.3 |
| **Database** | PostgreSQL (Neon Cloud) + SQLite (fallback local) |
| **Authentication** | JWT (python-jose), bcrypt password hashing |
| **Frontend** | Vanilla JS, HTML5, CSS3 — không framework |
| **APIs** | Spotify Web API (metadata/art), YouTube (playback) |
| **Deployment** | Docker (multi-stage), Hugging Face Spaces |
| **Model Storage** | Google Drive + gdown (tải tự động khi deploy) |

---

## 📊 Data Visualizations

<p align="center">
  <img src="visualizations/spotify_emotion_audio_radar.png" width="48%" />
  <img src="visualizations/tracks_features_hexbin_density.png" width="48%" />
</p>
<p align="center">
  <em>(Trái: Acoustic DNA của từng cảm xúc. Phải: Hexbin density map 1.2M bài hát)</em>
</p>

---

## 🚀 Hướng Dẫn Chạy Local (Development)

### Yêu Cầu Hệ Thống

- Python **3.11+**
- Git
- RAM: tối thiểu **8GB** (model loading)
- Dung lượng: tối thiểu **2GB** trống (model files)

### Bước 1: Clone Repository

```bash
git clone https://github.com/<your-username>/music_recommender.git
cd music_recommender
```

### Bước 2: Tạo Virtual Environment

```bash
# Windows
python -m venv .venv311
.venv311\Scripts\activate

# Linux / macOS
python3.11 -m venv .venv311
source .venv311/bin/activate
```

### Bước 3: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Lưu ý:** Package `torch==2.1.0` có thể mất 5-10 phút để tải (~2GB).

### Bước 4: Cấu Hình Biến Môi Trường

Tạo file `.env` ở thư mục gốc với nội dung sau:

```env
# Spotify API — lấy tại https://developer.spotify.com/dashboard
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here

# Database — có thể dùng SQLite local hoặc PostgreSQL cloud
# SQLite (không cần cài gì thêm):
DATABASE_URL=sqlite:///./web_app/music_rec.db

# PostgreSQL (tùy chọn, dành cho production):
# DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require

# JWT — đổi key này trong production!
JWT_SECRET_KEY=your-secret-key-change-this-in-production
JWT_EXPIRE_MINUTES=10080
```

### Bước 5: Tải Model Files

Model files quá lớn cho GitHub (>100MB). Có 2 cách:

**Cách A — Tự động (khuyến nghị):**

```bash
cd web_app
python download_helper.py
```

Script sẽ tự động tải tất cả files cần thiết từ Google Drive (~600MB).

**Cách B — Thủ công:**
Tải các files sau và đặt đúng vị trí:

| File | Vị trí | Kích thước |
|---|---|---|
| `best_model.pth` | `models/best_model.pth` | ~5MB |
| `faiss_index.bin` | `models/faiss_index.bin` | ~135MB |
| `faiss_index.bin.mappings.pkl` | `models/faiss_index.bin.mappings.pkl` | ~71MB |
| `audio_stats.json` | `models/audio_stats.json` | ~1KB |
| `tracks_faiss.index` | `models/tracks_faiss.index` | ~133MB |
| `tracks_faiss.index.mappings.pkl` | `models/tracks_faiss.index.mappings.pkl` | ~104MB |

### Bước 6: Khởi Chạy Ứng Dụng

```bash
# Từ thư mục gốc của project
python -m uvicorn web_app.app:app --host 0.0.0.0 --port 8000 --reload
```

Sau đó mở trình duyệt và truy cập: **<http://localhost:8000>**

### Kiểm Tra Sức Khỏe (Health Check)

```bash
curl http://localhost:8000/api/health
```

Response mong đợi:

```json
{
  "status": "ok",
  "model1_loaded": true,
  "model2_loaded": true
}
```

---

## 🔌 API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/api/health` | Kiểm tra trạng thái server & models |
| `GET` | `/api/search?q={query}` | Tìm kiếm bài hát (autocomplete) |
| `POST` | `/api/recommend/text` | Gợi ý theo tên bài hát hoặc mood text |
| `POST` | `/api/recommend/image` | Gợi ý theo ảnh upload (Model 1 + CLIP) |
| `POST` | `/api/recommend/image-v2` | Gợi ý theo ảnh upload (Model 2 + CLIP) |
| `POST` | `/api/spotify-enrich` | Lấy metadata Spotify cho danh sách bài hát |
| `POST` | `/auth/register` | Đăng ký tài khoản mới |
| `POST` | `/auth/login` | Đăng nhập, nhận JWT token |
| `GET` | `/api/history` | Lịch sử tìm kiếm của user (cần auth) |

---

## 🐳 Chạy Bằng Docker

### Build & Run Local

```bash
# Build image
docker build -t music-recommender .

# Run container
docker run -p 7860:7860 \
  -e SPOTIFY_CLIENT_ID=your_id \
  -e SPOTIFY_CLIENT_SECRET=your_secret \
  -e DATABASE_URL=sqlite:////data/music_rec.db \
  -e JWT_SECRET_KEY=your_secret_key \
  -v $(pwd)/models:/app/models \
  music-recommender
```

Truy cập: **<http://localhost:7860>**

### Docker Compose (khuyến nghị cho local)

Tạo file `docker-compose.yml`:

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "7860:7860"
    environment:
      - SPOTIFY_CLIENT_ID=${SPOTIFY_CLIENT_ID}
      - SPOTIFY_CLIENT_SECRET=${SPOTIFY_CLIENT_SECRET}
      - DATABASE_URL=sqlite:////data/music_rec.db
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    volumes:
      - ./models:/app/models
      - app_data:/data
volumes:
  app_data:
```

```bash
docker-compose up --build
```

---

## ☁️ Deploy lên Hugging Face Spaces

Project này được cấu hình sẵn để deploy lên **Hugging Face Spaces** với Docker SDK.

### Chuẩn Bị

1. Tạo tài khoản tại [huggingface.co](https://huggingface.co)
2. Tạo một Space mới → chọn **Docker** SDK
3. Cài Hugging Face CLI:

   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

### Cấu Hình Secrets trên HF Spaces

Vào **Space Settings → Repository secrets** và thêm:

| Secret Key | Giá trị |
|---|---|
| `SPOTIFY_CLIENT_ID` | Client ID từ Spotify Developer |
| `SPOTIFY_CLIENT_SECRET` | Client Secret từ Spotify Developer |
| `DATABASE_URL` | PostgreSQL connection string |
| `JWT_SECRET_KEY` | Secret key bất kỳ (random string dài) |

### Deploy

```bash
# Thêm HF Spaces remote
git remote add space https://huggingface.co/spaces/<username>/<space-name>

# Push nhánh hf-deploy lên space/main
git push space hf-deploy:main --force
```

> 📌 **Nhánh deploy:** Code đang chạy trên HF Spaces là nhánh `hf-deploy`, được push vào `space/main`.

### Quy Trình Download Models Tự Động

Khi app khởi động trên HF Spaces, `download_helper.py` sẽ tự động kiểm tra và tải các model files từ Google Drive nếu chúng chưa có. Quá trình này có thể mất **10-15 phút** lần đầu tiên.

---

## 🔁 Training Lại Các Mô Hình (Tùy Chọn)

### Train Model 1 (Hybrid BERT)

```bash
# 1. Chuẩn bị dữ liệu
python prepare_data.py

# 2. Training
python train_improved.py

# 3. Build FAISS index
python build_faiss_index.py
```

### Train Model 2 (Audio Autoencoder)

```bash
# 1. Chuẩn bị dữ liệu tracks
python -m audio_model.prepare_tracks_data

# 2. Training autoencoder
python -m audio_model.train_audio_model

# 3. Build FAISS index (1.2M songs)
python -m audio_model.build_audio_faiss
```

> ⚠️ Training đầy đủ yêu cầu GPU và có thể mất vài giờ.

---

## 🧪 Chạy Tests

```bash
# Tất cả tests
pytest tests/ -v

# Chỉ integration tests
pytest tests/integration/ -v

# Với coverage report
pytest tests/ --cov=web_app --cov-report=html
```

---

## ⚙️ Cấu Hình Nâng Cao

### Thay Đổi Google Drive Links (cho người fork)

Nếu bạn fork project này, cần upload model files lên Google Drive của bạn và cập nhật links trong `web_app/download_helper.py`:

```python
FILE_CONFIGS = {
    "model": {
        "url": "https://drive.google.com/uc?id=YOUR_FILE_ID",
        # ...
    },
    # ...
}
```

**Cách lấy File ID từ Google Drive:**

1. Upload file lên Google Drive
2. Right-click → Share → Copy link
3. Link dạng: `https://drive.google.com/file/d/FILE_ID_HERE/view`
4. Copy phần `FILE_ID_HERE`
5. Đổi quyền thành **"Anyone with the link can view"**

### Cấu Hình Database

- **SQLite** (local/dev): `DATABASE_URL=sqlite:///./web_app/music_rec.db`
- **PostgreSQL** (production): `DATABASE_URL=postgresql://user:pass@host/db?sslmode=require`
- Dịch vụ PostgreSQL miễn phí: [Neon](https://neon.tech), [Supabase](https://supabase.com)

---

## 🐛 Troubleshooting

### Lỗi "FAISS dimension mismatch"

```
Error: Index dimension (32) != query dimension (8)
```

**Nguyên nhân:** File `tracks_faiss.index` là bản cũ 32D.  
**Giải pháp:** Rebuild index: `python -m audio_model.build_audio_faiss`

### Lỗi "bcrypt detect_wrap_bug ValueError"

**Nguyên nhân:** `passlib 1.7.4` không tương thích với `bcrypt >= 4.0`.  
**Giải pháp:** Đảm bảo `requirements.txt` có: `bcrypt>=3.0,<4.0`

### Lỗi "gdown: fuzzy extraction failed"

**Nguyên nhân:** gdown < 5.2.0 không hỗ trợ Google Drive sharing links mới.  
**Giải pháp:** `pip install "gdown>=5.2.0"`

### App khởi động chậm (2-3 phút)

**Bình thường.** FAISS index lớn (~270MB tổng) cần thời gian tải vào RAM.  
Health check được cấu hình `--start-period=120s` để chờ quá trình này.

---

## 📜 Cấu Trúc Nhánh Git

| Nhánh | Mục đích |
|---|---|
| `main` | Development, tính năng mới nhất |
| `deployment` | Production fixes (Railway → HF) |
| `hf-deploy` | **Nhánh đang chạy trên Hugging Face Spaces** |

---

## 📄 License

Dự án này được phát triển cho mục đích học thuật (Đồ án Tốt nghiệp).

---

<div align="center">
  <p>Built with ❤️ using FastAPI + PyTorch + FAISS</p>
</div>
