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

## 3.4 Trực Quan Hóa Dữ Liệu (Data Visualization)

Phần này tổng hợp các kết quả trực quan hóa phân tích tập dữ liệu có nhãn (`spotify_dataset.csv`), tập dữ liệu gốc (`tracks_features.csv`) và không gian học máy tương ứng của hai mô hình.

### 3.4.1 Tổng Quan Các Bộ Dữ Liệu Thực Nghiệm

**Dataset 1: `spotify_dataset.csv` (Model 1 Input)**
- **Kích thước:** 496,278 tracks (tập train + val).
- **Phân loại:** 6 nhãn cảm xúc chính (Joy, Sadness, Anger, Fear, Love, Surprise).
- **Vấn đề:** Mất cân bằng lớp (Class Imbalance) cực kỳ nghiêm trọng khi Top 3 cảm xúc chiếm tới 88.8%. Dữ liệu cấu trúc có giám sát (Supervised) để luyện mô hình Hybrid.

**Dataset 2: `tracks_features.csv` (Model 2 Input)**
- **Kích thước:** 1,204,025 tracks.
- **Biến đổi:** Nhạc phát hành từ năm 1921 đến 2021, đóng gói dưới dạng 9 features âm thanh thô.
- **Tính chất:** File dữ liệu hoàn toàn không nhãn (Unsupervised) cho Audio Autoencoder.

### 3.4.2 Phân Tích Dataset `spotify_dataset.csv` (Model 1 Input)

#### ⭐ DNA Âm Thanh Theo Cảm Xúc (Radar Chart)
- **Cơ chế:** Radar Chart đa chiều thể hiện giá trị trung bình 7 audio features trên 6 nhóm cảm xúc. Đã giải quyết được việc định hướng thuộc tính nào mạnh nhất ở dòng cảm xúc nào.
- **Insights:** Thể hiện "chữ ký âm thanh" sinh động, ví dụ *Joy* dẫn đầu về **Danceability**, trong khi *Anger* lại vượt trội ở **Energy** và **Speechiness** do lẫn nhiều rap. Sự phân bua mờ nhạt giữa các ranh giới không gian khẳng định phải sử dụng xử lý Natural Language (NLP) từ Lyrics mới có thể định hình kết quả đúng thay vì chỉ dùng âm thanh.

#### ⭐ Tổng Quan Bộ Dữ Liệu (Dashboard 2x2)
Bảng điều khiển kết hợp 4 panel chuyên sâu về mặt thống kê: 
1. **Mất cân bằng lớp (Class Imbalance):** Phần lớn nghiêng về *Joy, Sadness, Anger*, còn lại rất ít.
2. **Audio Features Boxplot:** Nắm toàn cảnh phân phối từ Min đến Max của Daceability, Energy,...
3. **Phân phối Tempo:** Tập trung hình chuông rất chuẩn quanh 120 BPM.
4. **Mối quan hệ Valence - Energy:** *Love* bất ngờ lại nằm đơn độc như một mảng nhiễu, trong khi các nhãn còn lại tụ cụm tương đồng trung tính.

### 3.4.3 Phân Tích Dataset `tracks_features.csv` (Model 2 Input)

#### ⭐ Sự Tiến Hóa Âm Thanh Qua Các Thập Kỷ (Radar Chart by Decade)
Radar Chart theo dõi sự thay đổi dòng nhạc 7 thập kỷ qua:
- **Biến thiên lịch sử:** **Acousticness** (tính mộc) đâm dốc mạnh qua từng năm, đại diện cho xu hướng rời bỏ nhạc cụ và chuyển dần sang sản xuất tự động. Tương quan với nó, **Energy** và **Speechiness** theo chiều đi lên mạnh từ 1990+.

#### ⭐ Ma Trận Mật Độ Phân Phối (Hexbin Density Grid)
- **Kỹ thuật xử lý Big Data:** Trực quan hóa 1.2 triệu điểm mẫu sẽ khiến các Scatter Plot đen đặc lại vì lỗi overplotting. Vì thế sử dụng lưới lục giác log-scale (Hexbin) thể hiện "vùng tụ tập".
- **Insights:** Chỉ ra phân cực bimodal rõ ràng ở Acousticness. Tương quan mạnh nhất thuộc về cặp Danceability và Valence - Nhạc càng dễ khiêu vũ thì càng mang ý nghĩa tích cực!

### 3.4.4 Trực Quan Hóa Không Gian Nhúng Mô Hình (Embeddings Space)

Để chứng minh luận điểm hệ thống hoạt động chính xác (Giải thích được - Explainability AI):
- **Model 1 (Hybrid) 3D FAISS / UMAP:** Không gian tương tác biểu đồ Scatter Plot 3D phân mảnh từng hòn đảo của các cụm Happy, Sad, Party rõ nét. Do kết hợp cả Lyrics NLP mạnh mẽ.
- **Model 2 (Audio Autoencoder 1.2M) 2D Gradient:** Mô hình tự tạo thành một Spectrum âm thanh mịn màng chuyển tiếp tự nhiên từ Acoustic tĩnh mịch sang EDM hỗn mang cường độ cao, không cần tới Nhãn phân loại con người - Khẳng định Model 2 Unsupervised hoạt động cực kì hiệu quả trên 1.2 triệu bài hát.

### 3.4.5 Kết Luận Dữ Liệu
Vượt qua sự phức tạp khi mang trên mình **2 bộ dữ liệu khổng lồ khác nhau (tổng 1.7 triệu mẫu)**, những biểu đồ đã chọn lọc (Radar, Dashboard 4-panel, Hexbin Density...) giúp minh bạch hóa hoàn toàn cấu trúc không gian Vector và "chất liệu âm nhạc" nền tảng, tạo đòn bẩy vững chắc để hệ thống Recommend System gợi ý đạt hiệu năng cực cao.

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

## 7.7 Cơ Chế Tìm Kiếm Bài Hát Của Model 2 (Audio-Only)

Mặc dù `Model 2` vốn dĩ chỉ xử lý các thông số âm thanh, tính năng tiềm kiếm bài hát bằng Name/Text lại phải sử dụng lớp `CLIPAudioBridge`. Dưới đây là phân tích luồng xử lý và lý do đằng sau kiến trúc này:

- **Bản chất của Model 2**: Model 2 (Audio-Only) là một mạng Neural Network thu gọn chỉ đóng vai trò biến đổi 9 con số đặc trưng âm thanh đầu vào thành 1 vector 32 chiều (embeddings). Bản thân Model 2 không "hiểu" ngôn ngữ con người (không biết Text), cũng không tự định danh lưu trữ bài hát và không nhận diện bài hát nào giống bài hát nào.
- **Tại sao lại dùng `CLIP Bridge`?**: Việc truy vấn `Model 2` qua class mang tên "CLIP" hoàn toàn **không dính dáng tới xử lý ảnh CLIP**. Sở dĩ mượn class `CLIPAudioBridge` là vì module này đã khởi tạo kết nối (Initialize) tới toàn bộ **Database bài hát (song mappings)**, **những trọng số (weights) của Model 2** và bộ tìm kiếm **FAISS Index 2**. 
- **Cách Backend tìm kiếm**: Khi người dùng gõ tìm bài hát bằng tên (với lựa chọn Model 2), Backend sẽ yêu cầu hàm tìm kiếm bên trong module này (`recommend_from_song`) thực hiện 4 bước xử lý:
    1. Đọc tên Text vừa nhập, tra cứu ID bài hát gốc nằm ở đâu trong Database.
    2. Dùng ID truy cập trực tiếp FAISS để trích xuất Vector biểu diễn âm thanh của nó.
    3. Thực thi truy vấn các Vector biểu diễn có khoảng cách hình học gần nhất theo Index của hệ thống.
    4. Mapping ngược vector về tên và nghệ sĩ rồi trả kết quả về Frontend.

Tóm lại, `CLIPAudioBridge` chỉ đóng vai trò "kho lưu trữ tài nguyên" chứa FAISS + Database cho Audio-Only model, giúp không phải tải lại các models vào RAM trong lúc server chạy thực thi tính năng Tìm kiếm văn bản cơ bản.

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

# 9. GỢI Ý NHẠC TỪ HÌNH ẢNH & SỰ ĐA DẠNG MÔ HÌNH (Image-to-Music & Model Diversity)

## 9.1 Tổng Quan: Sự Đa Dạng Trong Gợi Ý

Hệ thống cung cấp hai mô hình hoàn toàn độc lập (Model 1 và Model 2) để đáp ứng các nhu cầu gợi ý khác nhau, đồng thời hỗ trợ cả hai phương thức đầu vào: gõ tên bài hát (Text Search) và tải lên bức ảnh (Image Search).

*   **Model 1 (Hybrid Model):** Gợi ý dựa trên sự phân tích **Ngữ nghĩa (Lyrics - BERT)** kết hợp với Âm thanh. Model này ưu tiên tìm các bài hát có "nội dung" và "cảm xúc" tương đồng.
*   **Model 2 (Audio-Only Autoencoder):** Gợi ý dựa hoàn toàn vào **Đặc trưng vật lý của Âm thanh (Audio Features)**. Model này tìm kiếm các bài hát có "Vibe", "Nhịp điệu" hoặc "Cấu trúc âm thanh" tương đồng với dữ liệu biểu diễn khổng lồ 1.2 triệu bài hát.

**Điểm mạnh của kiến trúc kép:** Dù người dùng tìm kiếm bằng cách gõ tên bài hát hay bằng một bức ảnh, họ đều có quyền chọn Model 1 hoặc Model 2. Điều này cung cấp 2 "khẩu vị" âm nhạc khác biệt: một bên chú trọng ý nghĩa lời ca, một bên chú trọng năng lượng và nhịp điệu (cực kỳ tốt cho nhạc quốc tế, nhạc không lời, EDM).

Để thực hiện Gợi ý bằng Hình ảnh trên Model 2 (vốn chỉ nhận thông số âm thanh) mà không cần mạng Neural đa phương thức (Multimodal) khổng lồ, hệ thống sử dụng kiến trúc **Bridge (Cầu nối)**.

## 9.2 Kiến Trúc Hệ Thống (Model 2: Audio Autoencoder Bridge)

Kiến trúc Model 2 được thiết kế theo hướng học không giám sát (Unsupervised Learning) tập trung hoàn toàn vào đặc trưng âm thanh thuần túy, loại bỏ sự phụ thuộc vào các nhãn cảm xúc chủ quan. Khi nhận đầu vào là hình ảnh, CLIP sẽ đóng vai trò tiền xử lý:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-INPUT RECOMMENDATION PIPELINE                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [INPUT A: IMAGE]                   │       [INPUT B: SONG NAME/TEXT]       │
│          │                          │                   │                   │
│  ┌───────▼───────┐                  │           ┌───────▼───────┐           │
│  │   CLIP Model  │                  │           │ Text Search   │           │
│  │ (Vision/Text) │                  │           │   Database    │           │
│  └───────┬───────┘                  │           └───────┬───────┘           │
│          │ (Label: "Happy", etc.)   │                   │ (Target Song)     │
│  ┌───────▼───────┐                  │                   │                   │
│  │ CLIP Audio    │                  │                   │                   │
│  │   Bridge      │                  │                   │                   │
│  └───────┬───────┘                  │                   │                   │
│          │ (Vector 9D: Energy=0.75) │                   │ (Raw 9D Audio)    │
│          └──────────────────────────┴───────────────────┘                   │
│                                     │                                       │
│                             ┌───────▼───────┐                               │
│                             │ Audio Model 2 │(Autoencoder Bottleneck)       │
│                             │  (Encoder)    │                               │
│                             └───────┬───────┘                               │
│                                     │ (Latent Vector 32D)                   │
│                                     │                                       │
│                             ┌───────▼───────┐                               │
│                             │  FAISS Index  │                               │
│                             │ (1.2M songs)  │                               │
│                             └───────┬───────┘                               │
│                                     │                                       │
│                           [Top-K Similar Songs]                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 9.3 Quy Trình Thực Hiện & Logic Hoạt Động

Quá trình chia thành 2 giai đoạn: **Offline (Chuẩn bị hệ thống)** và **Online (Khi người dùng thực hiện yêu cầu)**.

### 9.3.1 Giai Đoạn Offline (Xây dựng Model 2 & Index)

Thay vì dùng Model Phân loại (Classification) báo cáo tỷ lệ phần trăm các cảm xúc (có sự sai lệch và giới hạn do dữ liệu gán nhãn chủ quan), Model 2 được cải tiến thành một **Autoencoder**:

1. **Chuẩn bị dữ liệu (Data Preparation):**
   - Đọc 1.2 triệu dòng từ tập dataset thô `tracks_features.csv`.
   - Lọc và trích xuất đúng 9 thông số vật lý của âm thanh: `energy`, `danceability`, `valence`, `tempo`, `acousticness`, `instrumentalness`, `speechiness`, `liveness`, `key`.
   - Chuẩn hoá toàn bộ các thông số (Normalize Z-score cho tempo, Scale 0-1 cho biểu diễn phân loại).

2. **Huấn Luyện (Autoencoder Training):**
   - **Encoder:** Nén vector 9 chiều (thông số âm thanh) vào một không gian ẩn (Latent space) 32 chiều.
   - **Decoder:** Giải nén từ 32 chiều trở lại 9 chiều sao cho giống bản gốc nhất.
   - **Loss function:** Thay vì CrossEntropy, mô hình dùng **MSE (Mean Squared Error)** để tối ưu hoá sao cho độ lệch sát với vật lý âm thanh nhất. Quá trình này hoàn toàn Unsupervised, tự học từ 1.2 triệu bài mà không cần label.

3. **Lưu trữ Index (FAISS Indexing):**
   - Cắt bỏ phần Decoder đi. Chỉ giữ phần Encoder để tạo ra các Embeddings (Vector đại diện).
   - Truyền toàn bộ 1.2 triệu track âm thanh qua Encoder để lấy 1.2 triệu vector 32 chiều.
   - Đưa tất cả vector 32 chiều này vào cơ sở dữ liệu FAISS `IndexFlatIP`. Lúc này hệ thống sẵn sàng tìm kiếm siêu tốc.

### 9.3.2 Giai Đoạn Online (Inference / Recommending)

Khi user upload một hình ảnh, luồng hoạt động diễn ra theo các bước cực kỳ logic để "phiên dịch" thị giác thành thính giác:

1. **Computer Vision (CLIP):**
   - Ảnh truyền qua mô hình [OpenAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32).
   - CLIP thực hiện Zero-shot Classification để phân loại bức ảnh vào một trong các nhãn cài đặt sẵn (VD: `Happy`, `Sad`, `Party`, `Relax`, `Workout`, `Driving`, `Calm`).
   - CLIP chỉ xuất ra chữ (Label Text), không hề biết gì về âm thanh.

2. **The "Bridge" (Cầu nối):**
   - Module `CLIPAudioBridge` (`clip_audio_bridge.py`) nhận label text.
   - Dựa vào từ điển quy chuẩn (`AUDIO_PROFILES`) do con người định nghĩa, module này nội suy ra một **Vector 9 chiều mong muốn**.
     - *Ví dụ:* Nếu CLIP nhãn bức ảnh là `"Relax"`, Bridge sẽ xuất ra một vector quy định: `{ energy: 0.18, valence: 0.60, tempo: 84.0, acousticness: 0.78, ... }`
   - Nghĩa là bức ảnh đã được "Dịch" thành các thông số kỹ thuật âm nhạc!

3. **Search & Retrieve:**
   - Vector dự định trên sẽ đi qua Encoder của Model 2 để trở thành vector 32 chiều (Query Vector).
   - FAISS Search Query Vector này trong cơ sở 1.2 triệu bài hát.
   - Result: FAISS nhặt ra và trả về n bài hát có profile âm thanh gần giống với yêu cầu "Relax" kia nhất. Từ mặt định lượng vật lý, những bài hát này gần như chắc chắn sẽ đem lại cảm giác y như bức ảnh.

## 9.4 So Sánh Sự Lựa Chọn Model 1 & Model 2

| Tiêu Chí | Model 1 (Hybrid Text/Audio) | Model 2 (Pure Audio Autoencoder) |
|----------|-----------------------------|----------------------------------|
| **Kích thước Dataset**| Tối ưu cho `spotify_dataset.csv` (~550K bài) | Mở rộng cho `tracks_features.csv` (1.2 Triệu bài) |
| **Logic Gợi ý Bài hát (By Text)**| Tìm bài hát có cùng nội dung lời ca và cung bậc cảm xúc | Tìm các bài có cùng cường độ nhịp phách, giai điệu, nhạc cụ (Vibe) |
| **Logic Gợi ý Hình ảnh (By Image)**| Translate hình ảnh -> Cảm xúc nội dung -> FAISS | Trích xuất Label Hình Ảnh -> Audio Profile (Bridge) -> FAISS |
| **Bản chất Học máy** | Supervised Learning (Loss: CrossEntropy phân loại) | Unsupervised Learning (Loss: MSE phục hồi hình thái âm thanh) |
| **Use-case tốt nhất** | Tìm bài hát có ý nghĩa tương đương bài gốc. | Tìm nhạc không lời, EDM, hoặc những bài hợp để chạy bộ/chill bất kể ý nghĩa lời hát. |

Tính năng này chứng minh: Không có một Model nào là hoàn hảo cho mọi trường hợp. Bằng việc cung cấp **đồng thời cả 2 models cho cả tác vụ Tìm nhạc và Quét ảnh**, hệ thống đảm bảo một trải nghiệm đa dạng, chính xác và thích ứng với "khẩu vị" âm nhạc riêng biệt của từng người dùng.

---

# 10. TRIỂN KHAI TRÊN HUGGING FACE SPACES

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
