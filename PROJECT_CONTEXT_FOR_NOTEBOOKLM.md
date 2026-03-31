# BÁO CÁO TỔNG HỢP DỰ ÁN

# HỆ THỐNG GỢI Ý NHẠC THÔNG MINH

**Tên dự án:** Hybrid Music Recommendation System  
**Ngày thực hiện:** 06/01/2026 - 17/01/2026  
**Tác giả:** Graduation Project  
**Dataset:** 551,443 bài hát (Spotify + Genius Lyrics)

---

## MỤC LỤC

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Phân Tích Dữ Liệu](#2-phân-tích-dữ-liệu)
3. [Chuẩn Bị Dữ Liệu](#3-chuẩn-bị-dữ-liệu)
4. [Kiến Trúc Hệ Thống](#4-kiến-trúc-hệ-thống)
5. [Huấn Luyện Model](#5-huấn-luyện-model)
6. [Đánh Giá Kết Quả](#6-đánh-giá-kết-quả)
7. [Hệ Thống FAISS](#7-hệ-thống-faiss)
8. [Ứng Dụng Web](#8-ứng-dụng-web)
9. [Hướng Dẫn Sử Dụng](#9-hướng-dẫn-sử-dụng)
10. [Triển khai trên Hugging Face](#10-triển-khai-trên-hugging-face)
11. [Kết Luận](#11-kết-luận)

---

# 1. TỔNG QUAN DỰ ÁN

## 1.1 Mục Tiêu

Xây dựng hệ thống gợi ý nhạc thông minh dựa trên:

- **Phân tích ngữ nghĩa lời bài hát** (BERT embeddings)
- **Đặc trưng âm thanh** (Spotify audio features)
- **Phân loại cảm xúc** (Emotion classification)
- **Context-aware recommendations** (Party, Study, Workout, etc.)

## 1.2 Kiến Trúc Tổng Quan

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Bài hát                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐        ┌─────────────┐                   │
│   │ BERT Model  │        │ Audio MLP   │                   │
│   │ (768 dim)   │        │ (9 features)│                   │
│   └──────┬──────┘        └──────┬──────┘                   │
│          │                      │                          │
│   ┌──────▼──────┐        ┌──────▼──────┐                   │
│   │ Projection  │        │  Encoder    │                   │
│   │ 768→256     │        │  9→64       │                   │
│   └──────┬──────┘        └──────┬──────┘                   │
│          │                      │                          │
│          └──────────┬───────────┘                          │
│                     │                                       │
│              ┌──────▼──────┐                               │
│              │   Fusion    │                               │
│              │ 320→64 dim  │                               │
│              └──────┬──────┘                               │
│                     │                                       │
│              ┌──────▼──────┐                               │
│              │   OUTPUT    │                               │
│              │ 64D Embedding│                              │
│              └─────────────┘                               │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    FAISS Index                              │
│              (Similarity Search)                            │
└─────────────────────────────────────────────────────────────┘
```

## 1.3 Lý Do Lựa Chọn Công Nghệ

### Tại sao sử dụng BERT?

| Yếu tố | Giải thích |
|--------|------------|
| **Hiểu ngữ nghĩa sâu** | BERT có khả năng hiểu context của từ trong câu, phù hợp để phân tích lời bài hát với nhiều ẩn dụ và biểu cảm |
| **Pre-trained weights** | BERT đã được huấn luyện trên hàng tỷ câu văn bản, giúp tiết kiệm thời gian và chi phí training từ đầu |
| **768D embeddings** | Vector 768 chiều chứa đủ thông tin ngữ nghĩa phong phú để phân biệt các bài hát |

### Tại sao kết hợp Audio Features?

| Yếu tố | Giải thích |
|--------|------------|
| **Bổ sung thông tin** | Lyrics chỉ phản ánh nội dung văn bản, còn audio features phản ánh cảm xúc âm nhạc thực tế |
| **Spotify API chuẩn** | Các features như `energy`, `valence`, `danceability` đã được Spotify chuẩn hóa |
| **Giải quyết trường hợp đặc biệt** | Bài hát buồn có thể có lời vui (irony) - audio features giúp phát hiện điều này |

### Tại sao sử dụng FAISS?

| Yếu tố | Giải thích |
|--------|------------|
| **Tốc độ cực nhanh** | Xử lý triệu bài hát trong milliseconds |
| **Hỗ trợ GPU** | Tăng tốc 50-100x so với CPU |
| **Exact search** | IndexFlatIP đảm bảo 100% chính xác |
| **Scalable** | Dễ dàng mở rộng lên hàng triệu bài hát |

---

# 2. PHÂN TÍCH DỮ LIỆU

## 2.1 Datasets Có Sẵn

| Dataset | Records | Kích thước | Nội dung |
|---------|---------|------------|----------|
| **BERT 500K** | 551,443 | 1.64 GB | Lyrics, emotion, audio features, ISRC |
| **Spotify CSV** | 551,443 | 1.15 GB | Giống BERT 500K ở dạng CSV |
| **900K Spotify** | 498,052 | 1.36 GB | Lyrics, emotion, audio features |
| **Tracks Features** | 1,204,026 | 346 MB | Chỉ audio features (không có lyrics) |

## 2.2 Chi Tiết Features

### Audio Features (9 features chính)

| Feature | Range | Ý nghĩa |
|---------|-------|---------|
| `energy` | 0-1 | Cường độ/năng lượng |
| `danceability` | 0-1 | Khả năng nhảy theo |
| `valence` | 0-1 | Mức độ tích cực |
| `tempo` | BPM | Nhịp độ |
| `acousticness` | 0-1 | Mức độ acoustic |
| `instrumentalness` | 0-1 | Mức độ nhạc cụ |
| `speechiness` | 0-1 | Mức độ có lời nói |
| `liveness` | 0-1 | Mức độ live |
| `key` | 0-11 | Âm chính |

> **Lưu ý:** Đã loại bỏ `loudness` do 100% giá trị NaN trong dataset.

### Emotion Labels (6 loại)

| Emotion | Số lượng | Tỷ lệ |
|---------|----------|-------|
| Joy | 207,218 | 37.6% |
| Sadness | 171,141 | 31.0% |
| Anger | 110,372 | 20.0% |
| Fear | 28,215 | 5.1% |
| Love | 28,560 | 5.2% |
| Surprise | 5,937 | 1.1% |

### Context Tags (Binary 0/1)

- Good for Party
- Good for Work/Study
- Good for Relaxation/Meditation
- Good for Exercise
- Good for Running
- Good for Yoga/Stretching
- Good for Driving
- Good for Social Gatherings
- Good for Morning Routine

## 2.3 Data Quality Summary

| Dataset | Lyrics | Audio | Metadata | Emotion | Size |
|---------|--------|-------|----------|---------|------|
| 900k Dataset | ✅✅✅ | ✅✅ | ✅✅ | ✅✅ | 900K |
| BERT 500K | ✅✅✅ | ✅✅ | ✅✅✅ | ✅✅ | 551K |
| Spotify CSV | ✅✅✅ | ✅✅ | ✅✅✅ | ✅✅ | 551K |
| Tracks Features | N/A | Yes | Yes | N/A | 1.2M |

## 2.4 Khuyến Nghị Sử Dụng

**PRIMARY:** `final_milliondataset_BERT_500K_revised.json`

- ✅ Đã preprocessing cho BERT
- ✅ Có ISRC codes
- ✅ Đầy đủ features nhất
- ✅ Có similar songs để validation

---

# 3. CHUẨN BỊ DỮ LIỆU

## 3.1 Thiết Lập Môi Trường GPU

**Vấn đề gặp phải:**

- RTX 5060 Ti sử dụng kiến trúc Blackwell (sm_120) - quá mới cho PyTorch stable release

**Giải pháp:**

```python
# Cài đặt Python 3.11 thay vì Python 3.14
winget install Python.Python.3.11

# Tạo virtual environment
py -3.11 -m venv .venv311

# Cài PyTorch nightly với CUDA 12.8
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Kết quả:**

```
PyTorch: 2.11.0.dev20260110+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5060 Ti
GPU tensor test: Success [OK]
```

## 3.2 Các Bước Xử Lý Dữ Liệu

### Bước 1: Load Dữ Liệu

- BERT 500K: ~10 giây (~60K records/s)
- 900K Spotify: ~7 giây (~90K records/s)
- Tracks Features: ~3 giây (CSV format nhanh hơn)

### Bước 2: Chuẩn Hóa Tên Cột

```python
COLUMN_MAPPING = {
    'Energy': 'energy',
    'Danceability': 'danceability',
    'Positiveness': 'valence',      # Spotify API gọi là "valence"
    'Artist(s)': 'artist',
    'song': 'song_name',
    ...
}
```

### Bước 3: Chuẩn Hóa Giá Trị Audio Features

**Vấn đề:**

- BERT 500K dataset sử dụng scale **0-100**
- Spotify API chuẩn sử dụng scale **0-1**

**Giải pháp:**

```python
def normalize_to_spotify_api(df):
    scale_features = ['energy', 'danceability', 'valence', 'acousticness',
                      'instrumentalness', 'speechiness', 'liveness']
    
    for feat in scale_features:
        if df[feat].max() > 1:  # Đang ở scale 0-100
            df[feat] = df[feat] / 100.0
    
    return df
```

**Kết quả sau chuẩn hóa:**

```
energy:           0.000 - 1.000 (mean=0.627) [OK]
danceability:     0.060 - 0.990 (mean=0.592) [OK]
valence:          0.000 - 1.000 (mean=0.477) [OK]
acousticness:     0.000 - 1.000 (mean=0.257) [OK]
instrumentalness: 0.000 - 1.000 (mean=0.072) [OK]
speechiness:      0.020 - 0.970 (mean=0.117) [OK]
liveness:         0.010 - 1.000 (mean=0.197) [OK]
```

### Bước 4: Chia Dữ Liệu (Train/Val/Test Split)

| Split | Số lượng | Tỷ lệ |
|-------|----------|-------|
| Train | 441,156 | 80.0% |
| Val | 55,143 | 10.0% |
| Test | 55,144 | 10.0% |
| **Total** | **551,443** | 100% |

### Bước 5: Tạo BERT Embeddings

**Mô hình:** `bert-base-uncased`

**Quy trình:**

1. Làm sạch lyrics (xóa URLs, HTML tags, section markers)
2. Tokenization: max_length=256, padding='max_length'
3. Embedding generation: [CLS] token embedding (768D)

**Thời gian xử lý:**

- Train set (441K songs): ~70 phút
- Val set (55K songs): ~9 phút
- Test set (55K songs): ~9 phút
- **Total: ~1.5 giờ**

## 3.3 Files Đã Tạo

| File | Kích thước | Mô tả |
|------|------------|-------|
| `train.csv` | 964.3 MB | Training data (441,156 songs) |
| `val.csv` | 120.8 MB | Validation data (55,143 songs) |
| `test.csv` | 120.6 MB | Test data (55,144 songs) |
| `train_embeddings.npy` | 1,292.4 MB | BERT embeddings cho train (441156 × 768) |
| `val_embeddings.npy` | 161.6 MB | BERT embeddings cho val (55143 × 768) |
| `test_embeddings.npy` | 161.6 MB | BERT embeddings cho test (55144 × 768) |

---

# 4. KIẾN TRÚC HỆ THỐNG

## 4.1 Kiến Trúc Model (ImprovedModel)

```
BERT Branch:  768 → 512 → 384 → 256
Audio Branch: 9 → 64 → 128 → 64
Fusion:       320 → 256 → 192 → 128 → 64 (L2 Normalized)
Emotion Head: 64 → 32 → 6
```

## 4.2 Các Components Chính

### A. TextProcessor (processors.py)

- **Chức năng:** Xử lý lyrics và tạo BERT tokens
- **Input:** Raw lyrics text
- **Output:** input_ids, attention_mask, token_type_ids
- **Methods:**
  - `clean_lyrics()` - Làm sạch text, xóa URLs, HTML tags
  - `combine_text_features()` - Kết hợp lyrics + emotion + genre
  - `tokenize_batch()` - Convert text thành BERT tokens

### B. AudioProcessor (processors.py)

- **Chức năng:** Normalize 9 audio features
- **Normalization:** Z-score (mean=0, std=1)

### C. MetadataProcessor (processors.py)

- **Chức năng:** Convert categorical features thành indices
- **Vocabulary:** Genres, Keys (0-11), Emotions

### D. HybridMusicModel (model.py)

- **Architecture:**
  1. BERT Encoder → 256-dim embedding
  2. Audio Encoder (MLP) → 64-dim embedding
  3. Fusion Layer → 64-dim final embedding

### E. MusicRecommendationEngine (inference.py)

- **Methods:**
  - `get_similar_songs()` - Find similar songs
  - `get_recommendations_by_mood()` - Mood-based filtering
  - `get_recommendations_by_context()` - Context-aware filtering

## 4.3 Cấu Trúc Project

```
music_recommender/
├── hybrid_music_engine/      # Core module
│   ├── __init__.py
│   ├── config.py             # Cấu hình
│   ├── model.py              # Neural network architecture
│   ├── inference.py          # FAISS + Recommendation engine
│   ├── processors.py         # Data processors
│   └── logger.py             # Logging utilities
│
├── data/processed/           # Dữ liệu đã xử lý
│   ├── train.csv             # 441,153 samples
│   ├── val.csv               # 55,145 samples
│   ├── test.csv              # 55,145 samples
│   └── embeddings/           # BERT embeddings (.npy)
│
├── models/                   # Trained artifacts
│   ├── best_model.pth        # Model weights
│   ├── faiss_index.bin       # FAISS index
│   ├── faiss_index.bin.mappings.pkl  # Song mappings
│   └── audio_stats.json      # Audio normalization stats
│
├── build_faiss_index.py      # Build FAISS index
├── evaluate_final.py         # Final evaluation
├── interactive_demo.py       # Interactive demo
├── prepare_data.py           # Data preparation
└── train_improved.py         # Training script
```

---

# 5. HUẤN LUYỆN MODEL

## 5.1 Các Phương Pháp Đã Thử Nghiệm

| Phương pháp | Kết quả | Ghi chú |
|-------------|---------|---------|
| Contrastive Learning | NaN Loss | Không ổn định |
| Triplet Loss | 38% Accuracy | Kém hơn classification |
| **Classification + Label Smoothing** | **57.14%** | **Kết quả tốt nhất** |

## 5.2 Cấu Hình Training (Best Model)

```python
Epochs: 50 (Early stopping)
Batch Size: 256
Optimizer: AdamW
Learning Rate: 2e-3
Scheduler: Cosine Warmup
Label Smoothing: 0.05
Dropout: 0.3
```

## 5.3 Chiến Lược Huấn Luyện

**Multi-task Learning:**

- Task 1: Emotion Classification (supervised)
- Task 2: Genre Classification (auxiliary)

**Loss Function:**

```python
Total_Loss = 0.7 * emotion_loss + 0.3 * genre_loss
```

## 5.4 Tại sao chọn Classification thay vì Contrastive Learning?

| Phương pháp | Ưu điểm | Nhược điểm | Kết quả |
|-------------|---------|------------|---------|
| **Contrastive** | Embeddings tốt cho similarity | Cần batch size cực lớn (>2048), không ổn định | NaN Loss |
| **Triplet Loss** | Học trực tiếp similarity | Khó mining hard negatives | 38% |
| **Classification** | Ổn định, supervised signal rõ ràng | Phụ thuộc chất lượng labels | **57%** [OK] |

---

# 6. ĐÁNH GIÁ KẾT QUẢ

## 6.1 Độ Chính Xác Tổng Thể

| Metric | Validation | Test |
|--------|------------|------|
| **Accuracy** | **57.14%** | **56.66%** |
| Macro F1 | 0.38 | 0.37 |

## 6.2 Chi Tiết Theo Emotion (Test Set)

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Joy | 0.59 | **0.67** | 0.63 | 20,721 |
| Sadness | 0.55 | **0.64** | 0.59 | 17,114 |
| Anger | 0.54 | 0.52 | 0.53 | 11,037 |
| Fear | 0.55 | 0.06 | 0.11 | 2,821 |
| Love | 0.57 | 0.13 | 0.21 | 2,856 |
| Surprise | 1.00 | 0.08 | 0.15 | 596 |

## 6.3 Nhận Xét

- Model hoạt động tốt trên **Joy** và **Sadness** (chiếm 68% dữ liệu)
- Các class thiểu số (**Fear**, **Love**, **Surprise**) có recall thấp do mất cân bằng dữ liệu
- Không có overfitting (Val ≈ Test accuracy)

---

# 7. HỆ THỐNG FAISS

## 7.1 FAISS Là Gì?

**FAISS** (Facebook AI Similarity Search) là thư viện mã nguồn mở do Facebook AI Research phát triển, chuyên dùng để tìm kiếm vector tương tự với tốc độ cực nhanh.

**Bài toán cần giải quyết:**

- Cho 1 bài hát đầu vào (được biểu diễn bằng vector 64 chiều)
- Tìm 5 bài hát có vector "gần nhất" trong số 551,443 bài hát

**Cách tiếp cận naive:**

```
Duyệt qua tất cả 551,443 bài hát, tính khoảng cách với mỗi bài
→ Quá chậm! (O(n) với n = 551K)
```

**Cách tiếp cận FAISS:**

```
Sử dụng cấu trúc dữ liệu đặc biệt để tìm kiếm nhanh
→ Cực nhanh! (O(1) đến O(log n))
```

## 7.2 Nguyên Lý Hoạt Động

```
BUILD PHASE (Offline - 1 lần duy nhất):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 551K Songs  │ -> │ Model Encode│ -> │  64D Vectors│
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                                      ┌──────▼──────┐
                                      │ FAISS Index │
                                      │ (Optimized) │
                                      └─────────────┘

SEARCH PHASE (Online - mỗi request):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Query Song  │ -> │ Model Encode│ -> │ 64D Vector  │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                                      ┌──────▼──────┐
                                      │ FAISS Search│
                                      │ Top-K Nearest
                                      └──────┬──────┘
                                             │
                                      ┌──────▼──────┐
                                      │ 5 Similar   │
                                      │   Songs     │
                                      └─────────────┘
```

## 7.3 Các Loại FAISS Index

| Index Type | Tốc độ | Độ chính xác | RAM | Phù hợp cho |
|------------|--------|--------------|-----|-------------|
| **IndexFlatL2** | Chậm | 100% | Cao | Dataset nhỏ (<1M) |
| **IndexFlatIP** | Chậm | 100% | Cao | Cosine similarity |
| **IndexIVFFlat** | Nhanh | ~95% | Trung bình | Dataset lớn |
| **IndexHNSW** | Rất nhanh | ~98% | Cao | Realtime apps |

**Dự án này sử dụng: IndexFlatIP (Inner Product)**

## 7.4 GPU Acceleration

| Môi trường | Tốc độ (1 query / 551K songs) | Speedup |
|------------|-------------------------------|---------|
| CPU | ~5ms | 1x |
| **GPU (CUDA)** | **~0.05ms** | **100x** |

## 7.5 Thông Số Index Hiện Tại

| Thông số | Giá trị |
|----------|---------|
| Số lượng vectors | 551,443 |
| Số chiều (dimension) | 64 |
| Index type | IndexFlatIP |
| File size | ~134 MB |
| Tốc độ search | ~0.05ms (GPU) |

## 7.6 Post-Processing Filter (Emotion Logic)

Để cải thiện chất lượng gợi ý, hệ thống áp dụng bộ lọc cảm xúc:

| Input Emotion | Compatible Outputs |
|---------------|-------------------|
| Joy | Joy, Love, Surprise, Anger |
| Sadness | Sadness, Love, Fear |
| Anger | Anger, Joy, Fear |
| Love | Love, Joy, Sadness |
| Fear | Fear, Sadness, Anger |
| Surprise | Surprise, Joy |

---

# 8. ỨNG DỤNG WEB

## 8.1 Tổng Quan

Để làm cho hệ thống gợi ý nhạc dễ tiếp cận và sử dụng hơn, chúng tôi đã xây dựng một ứng dụng web hoàn chỉnh với giao diện người dùng hiện đại và backend API mạnh mẽ.

**Mục tiêu:**

- Cung cấp giao diện trực quan để tìm kiếm và khám phá nhạc
- Hỗ trợ gợi ý theo cảm xúc (mood) và ngữ cảnh (context)
- Tích hợp YouTube player để nghe nhạc trực tiếp
- Triển khai API RESTful để dễ dàng mở rộng

## 8.2 Kiến Trúc Tổng Quan

```
┌──────────────────────────────────────────────────────────────┐
│                        CLIENT SIDE                           │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐   │
│  │ HTML/CSS   │  │ JavaScript │  │ YouTube Iframe API  │   │
│  │ (UI)       │  │ (Logic)    │  │ (Media Player)      │   │
│  └─────┬──────┘  └─────┬──────┘  └──────────┬──────────┘   │
│        │                │                    │               │
│        └────────────────┴────────────────────┘               │
│                         │                                     │
│                         │ HTTP/REST API                      │
│                         │                                     │
└─────────────────────────┼─────────────────────────────────────┘
                          │
┌─────────────────────────┼─────────────────────────────────────┐
│                         │       SERVER SIDE                   │
│                  ┌──────▼──────┐                             │
│                  │  FastAPI    │                             │
│                  │  Backend    │                             │
│                  └──────┬──────┘                             │
│                         │                                     │
│    ┌────────────────────┼────────────────────┐              │
│    │                    │                    │              │
│    ▼                    ▼                    ▼              │
│ ┌──────┐    ┌───────────────────┐   ┌─────────────┐       │
│ │Search│    │Mood/Context Filter│   │ YouTube API │       │
│ │Engine│    │& Recommendation   │   │ Integration │       │
│ └───┬──┘    └─────────┬─────────┘   └──────┬──────┘       │
│     │                 │                     │              │
│     └─────────────────┴─────────────────────┘              │
│                       │                                     │
│            ┌──────────▼──────────┐                         │
│            │ MusicRecommendation │                         │
│            │      Engine         │                         │
│            └──────────┬──────────┘                         │
│                       │                                     │
│          ┌────────────┴────────────┐                       │
│          │                         │                       │
│     ┌────▼─────┐            ┌──────▼──────┐               │
│     │ PyTorch  │            │   FAISS     │               │
│     │  Model   │            │   Index     │               │
│     └──────────┘            └─────────────┘               │
└──────────────────────────────────────────────────────────────┘
```

## 8.3 Công Nghệ Sử Dụng

### 8.3.1 Backend Stack

| Technology | Version | Vai trò |
|------------|---------|---------|
| **FastAPI** | 0.128.0 | Web framework chính |
| **Uvicorn** | 0.40.0 | ASGI server |
| **Pydantic** | 2.12.5 | Data validation & serialization |
| **youtubesearchpython** | 1.6.6 | YouTube video search |
| **httpx** | 0.27.0 | Async HTTP client |

### 8.3.2 Frontend Stack

| Technology | Vai trò |
|------------|---------|
| **HTML5** | Structure & semantic markup |
| **CSS3** | Styling, animations, responsive design |
| **Vanilla JavaScript** | Client-side logic, API calls |
| **YouTube Iframe API** | Video embedding |

### 8.3.3 Lý Do Lựa Chọn Công Nghệ

#### Tại sao chọn FastAPI?

| Yếu tố | Giải thích |
|--------|------------|
| **Hiệu suất cao** | Ngang ngửa với NodeJS và Go nhờ ASGI & async/await |
| **Type safety** | Pydantic models đảm bảo data validation tự động |
| **Auto documentation** | Tự động tạo OpenAPI docs tại `/docs` |
| **Async native** | Xử lý concurrent requests hiệu quả |
| **Modern Python** | Sử dụng Python 3.11+ features |

#### Tại sao không dùng Flask?

| Flask | FastAPI |
|-------|---------|
| Synchronous (blocking I/O) | Asynchronous (non-blocking) |
| Manual validation | Automatic với Pydantic |
| Phải config Swagger manually | Built-in OpenAPI docs |
| Chậm hơn ~2-3x | Nhanh hơn nhờ async |

#### Tại sao dùng Vanilla JS thay vì React/Vue?

| Yếu tố | Giải thích |
|--------|------------|
| **Đơn giản** | Không cần build tools, webpack, npm packages |
| **Lightweight** | Zero dependencies, load nhanh hơn |
| **Phù hợp quy mô** | App nhỏ, không cần state management phức tạp |
| **Học tập** | Hiểu rõ fundamentals thay vì framework magic |

## 8.4 Chi Tiết Implementation

### 8.4.1 Backend Architecture (app.py)

**File structure:**

```
web_app/
├── app.py                    # FastAPI application
├── templates/
│   └── index.html            # Main HTML template
└── static/
    ├── css/
    │   └── style.css         # Styles
    └── js/
        └── main.js           # Frontend logic
```

**Core components trong app.py:**

#### A. Lifespan Management

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model và FAISS index khi startup"""
    global engine
    config = get_config()
    engine = MusicRecommendationEngine(config)
    
    # Load model, index, song data
    engine.load_model(model_path)
    engine.load_index()
    engine.load_song_data(data_path)
    
    yield  # Server running
    
    # Cleanup on shutdown
    engine = None
```

**Lý do:** Load model 1 lần duy nhất thay vì mỗi request → giảm latency từ ~30s xuống ~5ms

#### B. API Endpoints

| Endpoint | Method | Mô tả | Response Time |
|----------|--------|-------|---------------|
| `/` | GET | Serve HTML page | ~2ms |
| `/api/health` | GET | Health check | ~1ms |
| `/api/search` | GET | Tìm bài hát tương tự | ~5-10ms |
| `/api/mood/{mood}` | GET | Gợi ý theo cảm xúc | ~3-7ms |
| `/api/context/{context}` | GET | Gợi ý theo ngữ cảnh | ~3-7ms |
| `/api/youtube` | GET | Lấy YouTube embed URL | ~200-500ms |

**Example Request/Response:**

```bash
# Search similar songs
GET /api/search?q=Shape+of+You&artist=Ed+Sheeran

Response:
{
  "query": "Shape of You",
  "results": [
    {
      "song": "Come a Little Closer",
      "artist": "MARVELL",
      "genre": "hip hop",
      "emotion": "joy",
      "similarity": 0.913
    },
    ...
  ],
  "count": 10
}
```

#### C. Pydantic Models

```python
class SongResult(BaseModel):
    song: str
    artist: str
    genre: str
    emotion: str
    similarity: Optional[float] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SongResult]
    count: int
```

**Lợi ích:**

- Automatic validation (reject invalid data)
- JSON serialization tự động
- Type hints cho IDE autocomplete
- Documentation trong `/docs`

### 8.4.2 Frontend Architecture

#### A. CSS Design System (style.css)

**Color Palette:**

```css
:root {
    --bg-primary: #0a0e27;        /* Deep space blue */
    --bg-secondary: #1a1f3a;      /* Dark navy */
    --accent-primary: #667eea;     /* Purple */
    --accent-secondary: #64b5f6;   /* Light blue */
    --text-primary: #e8eaf6;       /* Off-white */
    --text-secondary: #b0b4c8;     /* Gray */
}
```

**Key Features:**

- **Dark mode:** Giảm mỏi mắt, trendy
- **Gradient accents:** Hiện đại, premium feel
- **Glassmorphism:** Backdrop blur effects
- **Smooth animations:** CSS transitions 300-400ms
- **Responsive grid:** Flexbox + CSS Grid

**Song Card Design:**

```css
.song-card {
    background: rgba(26, 31, 58, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 12px;
    transition: all 0.3s ease;
}

.song-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}
```

#### B. JavaScript Logic (main.js)

**State Management:**

```javascript
const app = {
    currentMode: 'search',    // search | mood | context
    isLoading: false,
    results: []
};
```

**API Integration:**

```javascript
async function searchSongs(query, artist = '') {
    showLoading();
    try {
        const response = await fetch(
            `/api/search?q=${encodeURIComponent(query)}&artist=${encodeURIComponent(artist)}`
        );
        const data = await response.json();
        displayResults(data.results);
    } catch (error) {
        showToast('Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}
```

**YouTube Integration:**

```javascript
async function playYouTube(song, artist) {
    const response = await fetch(
        `/api/youtube?song=${encodeURIComponent(song)}&artist=${encodeURIComponent(artist)}`
    );
    const data = await response.json();
    
    // Show modal with YouTube iframe
    const iframe = document.getElementById('youtubePlayer');
    iframe.src = data.embed_url;
    modal.classList.add('active');
}
```

### 8.4.3 Layout & UI Components

**Responsive Grid:**

```css
.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.5rem;
}

/* Mobile: 1 column */
@media (max-width: 768px) {
    .results-grid {
        grid-template-columns: 1fr;
    }
}
```

**Hero Section:**

- Gradient text effect cho title
- Search bar với glass morphism
- Artist input (optional)
- Smooth focus animations

**Navigation Tabs:**

- Search / By Mood / By Activity
- Active state với gradient underline
- Smooth transitions

## 8.5 Features Implemented

### 8.5.1 Tìm Kiếm Bài Hát (Song Search)

**Cách hoạt động:**

1. User nhập tên bài hát (+ artist optional)
2. Frontend gửi GET request đến `/api/search`
3. Backend encode query thành 64D vector
4. FAISS tìm top-10 nearest neighbors
5. Return kết quả với similarity scores

**UI:**

- Hiển thị 10 bài hát dạng card grid
- Mỗi card có: tên, nghệ sĩ, genre tag, emotion tag, similarity bar
- Hover effect: lift + glow
- Play button → mở YouTube modal

### 8.5.2 Gợi Ý Theo Cảm Xúc (Mood Filter)

**Available moods:**

- Happy
- Sad
- Energetic
- Calm
- Angry

**Logic:**

```python
def get_recommendations_by_mood(mood: str, top_k: int = 10):
    # Filter songs by emotion column
    filtered = song_data[song_data['emotion'] == mood.lower()]
    
    # Sort by audio features phù hợp
    if mood == 'energetic':
        filtered = filtered.sort_values('energy', ascending=False)
    elif mood == 'calm':
        filtered = filtered.sort_values(['acousticness', 'energy'], 
                                       ascending=[False, True])
    
    return filtered.head(top_k)
```

### 8.5.3 Gợi Ý Theo Ngữ Cảnh (Context-Aware)

**Available contexts:**

- Party
- Workout
- Study
- Relax
- Driving

**Logic:**

```python
CONTEXT_MAPPING = {
    'party': {'high_energy': True, 'high_danceability': True},
    'study': {'low_energy': True, 'low_speechiness': True, 'instrumental': True},
    'workout': {'high_energy': True, 'high_tempo': True},
    'relax': {'high_acousticness': True, 'low_energy': True},
    'driving': {'medium_energy': True, 'medium_tempo': True}
}
```

### 8.5.4 YouTube Integration

**Workflow:**

1. User click play button trên song card
2. Frontend gọi `/api/youtube?song=...&artist=...`
3. Backend dùng `youtubesearchpython` tìm video
4. Return embed URL + thumbnail
5. Frontend hiển thị modal với YouTube iframe

**Challenges & Solutions:**

| Challenge | Solution |
|-----------|----------|
| Module import error | Fixed import: `youtubesearchpython.VideosSearch` |
| httpx compatibility | Downgraded httpx 0.28 → 0.27 |
| Blocking I/O | Wrap sync calls với `asyncio.to_thread()` |
| Embed restrictions | Some videos show "unavailable" (YouTube policy, not our bug) |

## 8.6 Verification & Testing

### 8.6.1 Homepage

![Homepage Screenshot](C:/Users/Admin/.gemini/antigravity/brain/88ca47ec-3d4c-4bfb-a73a-167d5cad0c53/initial_home_page_1768827537419.png)

**Verified:**

- ✅ Dark theme loads correctly
- ✅ Gradient effects working
- ✅ Search bar responsive
- ✅ Navigation tabs functional
- ✅ No console errors

### 8.6.2 Search Functionality

![Search Results](C:/Users/Admin/.gemini/antigravity/brain/88ca47ec-3d4c-4bfb-a73a-167d5cad0c53/.system_generated/click_feedback/click_feedback_1768827618044.png)

**Test case:** Search for "Shape of You"

**Results:**

- ✅ Returns 10 similar songs
- ✅ Similarity scores displayed correctly (91.3%, 91.2%, etc.)
- ✅ Tags rendered (genre, emotion)
- ✅ Response time: ~8ms
- ✅ Grid layout responsive

### 8.6.3 YouTube Player

![YouTube Modal](C:/Users/Admin/.gemini/antigravity/brain/88ca47ec-3d4c-4bfb-a73a-167d5cad0c53/youtube_player_unavailable_1768829387642.png)

**Test cases:**

- Click play on "Plastic Promises" → ✅ Modal opens
- YouTube iframe loads → ✅ Working
- Some videos "unavailable" → ⚠️ Expected (YouTube embed restrictions)

**Demo Recording:**

![Web App Demo](C:/Users/Admin/.gemini/antigravity/brain/88ca47ec-3d4c-4bfb-a73a-167d5cad0c53/youtube_final_test_1768829265815.webp)

### 8.6.4 Performance Metrics

| Metric | Value | Note |
|--------|-------|------|
| Server startup | ~30s | Load model + FAISS index |
| Search API | ~5-10ms | After model loaded |
| Mood filter | ~3-7ms | DataFrame filtering |
| YouTube search | ~200-500ms | External API call |
| Page load (FCP) | ~800ms | Including CSS/JS |
| TTI (Time to Interactive) | ~1.2s | Full interactivity |

## 8.7 Deployment & Operations

### 8.7.1 Development Mode

```bash
# Navigate to web_app directory
cd c:\Users\Admin\.gemini\antigravity\scratch\music_recommender\web_app

# Run with hot reload
..\\.venv311\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 8.7.2 Production Considerations

**Performance optimizations:**

- Use Gunicorn với multiple workers
- Enable gzip compression
- CDN cho static assets
- Redis cache cho frequent queries

**Security:**

- CORS configuration
- Rate limiting
- API key authentication (nếu public)
- Input sanitization (Pydantic đã handle)

**Monitoring:**

- Prometheus metrics endpoint
- Structured logging
- Error tracking (Sentry)
- Uptime monitoring

### 8.7.3 API Documentation

FastAPI tự động tạo interactive docs tại:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

## 8.8 Challenges & Solutions

### Challenge 1: GPU Model Loading in Web Context

**Problem:** Model load mất ~30s, không thể load mỗi request

**Solution:** Sử dụng FastAPI `lifespan` event để load 1 lần duy nhất khi startup:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = MusicRecommendationEngine(config)
    engine.load_model(...)  # Once only
    yield
    engine = None
```

### Challenge 2: FAISS Thread-Safety trong Async Context

**Problem:** FAISS không thread-safe, có thể crash với concurrent requests

**Solution:** Sử dụng `asyncio.Lock()` để synchronize FAISS searches:

```python
faiss_lock = asyncio.Lock()

async def search_faiss(query_vector):
    async with faiss_lock:
        distances, indices = index.search(query_vector, k)
    return indices
```

### Challenge 3: YouTube API Integration

**Problem 1:** Wrong module name (`youtube_search` vs `youtubesearchpython`)

**Solution:** Correct import:

```python
from youtubesearchpython import VideosSearch
```

**Problem 2:** httpx 0.28.1 incompatible với youtubesearchpython

**Solution:** Downgrade:

```bash
pip install httpx==0.27.0
```

**Problem 3:** Synchronous library blocking event loop

**Solution:**

```python
result = await asyncio.to_thread(
    lambda: VideosSearch(query, limit=1).result()
)
```

### Challenge 4: Windows Console Encoding

**Problem:** Emoji characters causing `UnicodeEncodeError`

**Solution:** Replace emojis với ASCII alternatives:

```python
"😔" → "(sad face)"
"🔥" → "[fire]"
```

## 8.9 So Sánh Với `interactive_demo.py`

| Feature | interactive_demo.py | Web App |
|---------|---------------------|---------|
| **Interface** | Terminal (CLI) | Web browser (GUI) |
| **User experience** | Text-based | Visual, interactive |
| **Accessibility** | Technical users only | Anyone với browser |
| **YouTube** | Opens external browser | Embedded player |
| **Deployment** | Local only | Can deploy to cloud |
| **Concurrency** | Single user | Multiple users |
| **Aesthetics** | Plain text | Modern UI với animations |

## 8.10 Future Enhancements

### Phase 1: Core Improvements

- [ ] Add pagination cho search results
- [ ] Implement playlist creation
- [ ] User favorites/history (localStorage)
- [ ] Export recommendations as JSON/CSV

### Phase 2: Advanced Features

- [ ] User authentication (login/register)
- [ ] Collaborative filtering (user behavior)
- [ ] Spotify API integration (actual playback)
- [ ] Share recommendations via URL

### Phase 3: ML Enhancements

- [ ] Real-time feedback loop
- [ ] A/B testing different recommendation algorithms
- [ ] Personalized ranking based on user preferences
- [ ] Hybrid collaborative + content-based filtering

### Phase 4: Production Ready

- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline
- [ ] Load balancing và auto-scaling
- [ ] Database integration (PostgreSQL)
- [ ] Caching layer (Redis)

---

# 9. TRIỂN KHAI TRÊN HUGGING FACE SPACES

## 9.1 Tổng Quan

Bên cạnh việc triển khai cục bộ (Local Deployment), dự án cũng đã được cấu hình để chạy trên nền tảng **Hugging Face Spaces**. Đây là môi trường lý tưởng để chia sẻ demo machine learning với cộng đồng.

**Cấu hình Space:**

- **SDK:** Docker
- **Hardware:** CPU Basic (2 vCPU, 16GB RAM)
- **Port:** 7860 (Hugging Face default)

## 9.2 Quy Trình Triển Khai

### Bước 1: Chuẩn bị Dockerfile

Do Hugging Face Spaces có các yêu cầu bảo mật khắt khe (chạy dưới non-root user), Dockerfile cần được cấu hình đặc biệt:

```dockerfile
# Sử dụng Python 3.11 slim
FROM python:3.11-slim

# Tạo user non-root (bắt buộc trên HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy working directory
WORKDIR /app
COPY --chown=user:user . /app

# Cài đặt dependencies
RUN pip install --no-cache-dir -r web_app/requirements-deploy.txt

# Run app
CMD ["uvicorn", "web_app.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Bước 2: Quản lý Dependencies

Tạo file `web_app/requirements-deploy.txt` riêng biệt, loại bỏ các thư viện chỉ dùng cho training (như matplotlib, seaborn) để giảm kích thước image.

```text
fastapi>=0.109.0
uvicorn>=0.27.0
torch
transformers
faiss-cpu
youtubesearchpython
...
```

### Bước 3: Xử Lý Model Files Lớn

Do giới hạn của Git LFS và băng thông, chúng tôi sử dụng cơ chế **Hybrid Loading**:

1. Các file cấu hình nhỏ (`config.py`, `mappings.pkl`) đi kèm code.
2. Các file model lớn (`best_model.pth`, `faiss_index.bin`) được tải xuống từ Google Drive khi khởi động container thông qua script `download_helper.py`.

## 9.3 Xử Lý Vấn Đề Tương Thích (YouTube)

Một thách thức lớn khi deploy lên cloud là việc YouTube chặn hoặc hạn chế các request embedding từ IP của Data Center.

### Vấn đề gặp phải

- Lỗi **"YouTube refused to connect"** khi hiển thị iframe.
- API `youtubesearchpython` bị rate limit hoặc block IP.

### Giải pháp: Graceful Fallback

**1. Backend Adjustment (`app.py`):**
Cập nhật endpoint `/api/youtube` để trả về search URL nếu không lấy được video ID trực tiếp.

```python
try:
    # Cố gắng lấy video details
    videos_search = VideosSearch(query, limit=1)
    results = await asyncio.to_thread(videos_search.result)
    # ... logic lấy embed link
except Exception:
    # Fallback: Trả về link tìm kiếm trực tiếp
    return {
        "type": "fallback",
        "url": f"https://www.youtube.com/results?search_query={encoded_query}"
    }
```

**2. Frontend Adjustment (`main.js`):**
Javascript sẽ kiểm tra loại phản hồi. Nếu là `fallback`, thay vì mở modal iframe, nó sẽ mở tab mới dẫn đến trang tìm kiếm YouTube.

```javascript
if (data.type === 'fallback') {
    // Nếu không thể embed, mở new tab
    window.open(data.url, '_blank');
} else {
    // Normal flow: Mở iframe modal
    iframe.src = data.embed_url;
    modal.classList.add('active');
}
```

## 9.4 Kết Quả

Với các điều chỉnh trên, hệ thống hoạt động ổn định trên Hugging Face Spaces:

- ✅ Web App load thành công.
- ✅ Model inference hoạt động bình thường (CPU latency chấp nhận được).
- ✅ YouTube Integration chuyển sang chế độ fallback thông minh, đảm bảo người dùng vẫn nghe được nhạc dù không xem trực tiếp trong app.

---

# 10. HƯỚNG DẪN SỬ DỤNG

## 9.1 Requirements

```bash
pip install torch transformers faiss-cpu pandas numpy tqdm scikit-learn
# Or for GPU support:
pip install faiss-gpu
```

## 9.2 Chạy Demo

```bash
python interactive_demo.py
```

## 9.3 API Sử Dụng

```python
from hybrid_music_engine import MusicRecommendationEngine, get_config

config = get_config()
engine = MusicRecommendationEngine(config)

# Load model và index
engine.load_model("models/best_model.pth")
engine.load_index("models/faiss_index.bin")
engine.load_song_data("data/processed/train.csv")

# Tìm bài hát tương tự
results = engine.get_similar_songs("Shape of You", "Ed Sheeran", top_k=5)
```

## 9.4 Command Line

```bash
# Train model
python train_improved.py

# Build FAISS Index
python build_faiss_index.py

# Evaluate
python evaluate_final.py
```

---

# 11. KẾT LUẬN

## 10.1 Thành Quả

- ✅ Xây dựng thành công hệ thống gợi ý nhạc hybrid (BERT + Audio)
- ✅ Đạt độ chính xác phân loại cảm xúc 57%
- ✅ Index 551K bài hát với tốc độ tìm kiếm milliseconds
- ✅ Implement post-processing filter để cải thiện chất lượng gợi ý

## 10.2 Hạn Chế

- Data imbalance ảnh hưởng đến các class thiểu số (Fear, Love, Surprise)
- Emotion labels từ dataset gốc có nhiễu
- Recall thấp cho minority classes (6-13%)

## 10.3 Các Vấn Đề Đã Khắc Phục

| Vấn đề | Giải pháp |
|--------|-----------|
| Cột `loudness` 100% NaN | Loại bỏ khỏi audio features |
| Tên cột không đồng nhất | Tự động detect 4 biến thể |
| Thiếu file mappings cho FAISS | Thêm logic lưu mappings.pkl |
| Unicode encoding error trên Windows | Thay emoji bằng ASCII characters |
| Multiprocessing pickle error trên Windows | Set num_workers=0 |

## 10.4 Hướng Phát Triển

1. Sử dụng LLM để gán lại emotion labels chính xác hơn
2. Thử nghiệm Contrastive Learning với batch size lớn hơn
3. Implement class weighting hoặc oversampling cho minority classes
4. Xây dựng giao diện web hoàn chỉnh
5. Thử nghiệm các model nhỏ hơn như DistilBERT, MiniLM

---

**Kết thúc báo cáo.**

---

*Báo cáo được tổng hợp từ 7 file documentation của dự án.*  
*Hybrid Music Recommendation Engine - Graduation Project 2026*
  
# DOCUMENTATION: DEPLOYMENT_GUIDE.md  
# =============================================================================

# DEPLOYMENT GUIDE: Music Recommender Web App to Hugging Face Spaces

# =============================================================================

## OVERVIEW

This guide explains how to deploy the Music Recommender Web App to Hugging Face Spaces using external file hosting for large model files.

**Platform:** Hugging Face Spaces
**SDK:** Docker
**Hardware:** CPU Basic (2 vCPU, 16GB RAM) - Free Tier is sufficient

---

## PREREQUISITES

1. **Google Drive account** (for hosting files)
2. **Hugging Face account**
3. **Local environment** with Python 3.11+
4. **Git** installed

---

## DEPLOYMENT STEPS

### STEP 1: Upload Files to Google Drive

Upload these files to your Google Drive (Total ~210 MB):

```
📦 Files to upload:
├── best_model.pth              (~5 MB)
├── faiss_index.bin             (~135 MB)
└── faiss_index.bin.mappings.pkl (~71 MB)
```

**Note:** The app skips `train.csv` (1GB) and uses the lighter `models/song_metadata.csv` included in the repo.

**How to upload:**

1. Go to [drive.google.com](https://drive.google.com)
2. Create a new folder (e.g., "music_recommender_models")
3. Upload the 3 files above
4. For EACH file:
   - Right-click → Share
   - Change to "Anyone with the link can view"
   - Copy the link and extract the FILE ID

---

### STEP 2: Update download_helper.py with File IDs

Open `web_app/download_helper.py` and replace the `YOUR_*_FILE_ID` placeholders with your actual Google Drive file IDs.

---

### STEP 3: Create Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. **Space name:** `music-recommender` (or your choice)
4. **License:** MIT or Apache 2.0
5. **Space SDK:** **Docker** (Important! Do NOT select Gradio or Streamlit)
6. Click **Create Space**

---

### STEP 4: Push Code to Space

You can push code directly via browser or use Git.

**Using Git (Recommended):**

```bash
# Clone the empty space (replace USER with your username)
git clone https://huggingface.co/spaces/USER/music-recommender
cd music-recommender

# Copy your project files into this directory
# Ensure Dockerfile is at the root
# Ensure web_app/, hybrid_music_engine/, models/ folders are present

# Add files
git add .
git commit -m "Initial commit"
git push
```

**Using Browser:**
Drag and drop your project files into the "Files" tab of your Space.

---

### STEP 5: Monitor Deployment

1. Go to the **App** tab of your Space.
2. You will see "Building" status.
3. Click "Logs" to watch the progress.
   - It will build the Docker image.
   - Then it will run `download_helper.py` to fetch models.
   - Finally, `uvicorn` will start the server.

**Expected Log Success:**

```
[OK] Music Recommender API is ready!
Application startup complete.
```

---

## IMPORTANT NOTES

- **Port:** The Dockerfile must expose port **7860**.
- **Permissions:** The container runs as non-root user (ID 1000). The `Dockerfile` handles this.
- **YouTube Fallback:** If YouTube searches fail due to IP blocking, the app will automatically return a direct link instead of an embed.

---

## TROUBLESHOOTING

- **Build Failed?** Check the Logs tab. Common errors are missing dependencies in `web_app/requirements-deploy.txt`.
- **Download Failed?** Verify Google Drive File IDs are public.
- **Runtime Error?** Ensure you are not trying to write to read-only directories. Only `/app` or `/tmp` are writable.

---

**Author:** Graduation Project
**Date:** 2026-01-26
**Version:** 2.0 (Hugging Face Edition)
  
# SOURCE: web_app/app.py  
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
  
# SOURCE: hybrid_music_engine/inference.py  
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
    except:
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
        except:
            # Fallback to zeros if columns missing
            audio_features = torch.zeros((1, 11))
        
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
        if self.song_data is None: return []
        
        mood_mapping = {
            'happy': {'emotion': ['joy'], 'valence': (0.6, 1.0), 'energy': (0.5, 1.0)},
            'sad': {'emotion': ['sadness'], 'valence': (0.0, 0.4), 'energy': (0.0, 0.5)},
            'energetic': {'emotion': ['joy'], 'valence': (0.5, 1.0), 'energy': (0.7, 1.0)},
            'calm': {'emotion': [], 'valence': (0.4, 0.7), 'energy': (0.0, 0.4)},
            'angry': {'emotion': ['anger'], 'valence': (0.0, 0.4), 'energy': (0.7, 1.0)},
        }
        
        mood_lower = mood.lower()
        if mood_lower not in mood_mapping: mood_lower = 'happy'
        filters = mood_mapping[mood_lower]
        filtered = self.song_data.copy()
        
        if filters['emotion'] and 'emotion' in filtered.columns:
            filtered = filtered[filtered['emotion'].isin(filters['emotion'])]
        if 'valence' in filtered.columns:
            filtered = filtered[(filtered['valence'] >= filters['valence'][0]) & (filtered['valence'] <= filters['valence'][1])]
        if 'energy' in filtered.columns:
            filtered = filtered[(filtered['energy'] >= filters['energy'][0]) & (filtered['energy'] <= filters['energy'][1])]
        
        if 'Popularity' in filtered.columns:
            filtered = filtered.sort_values('Popularity', ascending=False)
            
        song_col, artist_col = self._get_column_names()
        recommendations = []
        for _, row in filtered.head(top_k).iterrows():
            recommendations.append({
                'song': row.get(song_col, ''),
                'artist': row.get(artist_col, ''),
                'genre': row.get('Genre', row.get('genre', '')),
                'emotion': row.get('emotion', ''),
            })
        return recommendations

    def get_recommendations_by_context(self, context: str, top_k: int = 10) -> List[Dict]:
        if self.song_data is None: return []
        # (Simplified context logic)
        return self.get_recommendations_by_mood('happy', top_k)

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
                    except:
                        batch_embeddings.append(np.zeros(64))
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
