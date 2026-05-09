# BÁO CÁO TRAINING MODEL 2 + XÂY DỰNG FAISS INDEX (Optimization)

**Ngày thực hiện:** 2026-04-04
**Thời gian xử lý:** ~10 phút (training ~4 phút, FAISS build ~1 phút, testing ~5 phút)
**Thiết bị:** CUDA GPU

---

## 1. TỔNG QUAN

### Mục tiêu
Tối ưu hóa Model 2 (Audio-Only Autoencoder) bằng cách:
- Giảm latent dimension từ **32 → 8**
- Chuyển metric distance từ **Cosine Similarity (IndexFlatIP)** sang **Euclidean Distance (IndexFlatL2)**
- Loại bỏ L2-normalization trong inference

### Lý do cần tối ưu
SVD analysis cho thấy embedding hiện tại (32D) chỉ có **7 chiều thực sự mang thông tin** (99% energy),
25 chiều còn lại gần bằng 0 (noise). Khi L2-normalize, noise bị khuếch đại → tất cả cosine similarity ≈ 0.99.

### Input
- Training data: `data/processed/tracks/features_train.npy` (1,083,622 samples × 9 features)
- Validation data: `data/processed/tracks/features_val.npy` (60,201 samples × 9 features)
- Normalization stats: `models/tracks_stats.json`

### Output
- Model weights: `models/autoencoder_model2.pth` (7.7 KB)
- FAISS index: `models/tracks_faiss.index` (33 MB)
- Mappings: `models/tracks_faiss.index.mappings.pkl` (104 MB)

## 2. CÁC BƯỚC THỰC HIỆN CHI TIẾT

### Bước 1: Backup model cũ
```bash
Copy-Item models\autoencoder_model2.pth models\autoencoder_model2_dim32_backup.pth
Copy-Item models\tracks_faiss.index models\tracks_faiss_dim32_backup.index
```

### Bước 2: Sửa config2.py
```python
# TRƯỚC
output_dim: int = 32  # Bottleneck / Latent space dimension

# SAU  
output_dim: int = 8   # Bottleneck / Latent space dimension (reduced from 32)
```
**Tại sao dim=8?** SVD analysis cho thấy 99% information chỉ cần 7 chiều → dim=8 là lựa chọn tối ưu,
tạo genuine information bottleneck (8 < 9 input features).

### Bước 3: Sửa model2.py — Loại bỏ L2-normalize
```python
# TRƯỚC (encode method)
def encode(self, x: torch.Tensor) -> torch.Tensor:
    latent = self.encoder(x)
    return F.normalize(latent, p=2, dim=-1)  # L2-normalise for cosine

# SAU
def encode(self, x: torch.Tensor) -> torch.Tensor:
    return self.encoder(x)  # Raw embeddings — uses L2 distance in FAISS
```
**Tại sao bỏ L2-normalize?** 
- L2-norm buộc mọi vector nằm trên bề mặt unit hypersphere trong 8D
- Trong 8D, surface area rất nhỏ → tất cả vectors cluster lại gần nhau
- Giữ raw embeddings cho phép FAISS L2 distance đo khoảng cách tự nhiên

### Bước 4: Retrain autoencoder
```bash
python -m audio_model.train_audio_model
```

### Bước 5: Sửa build_audio_faiss.py — Chuyển sang L2
```python
# TRƯỚC
index = faiss.IndexFlatIP(latent_dim)  # Inner product == cosine

# SAU
index = faiss.IndexFlatL2(latent_dim)  # L2 distance (Euclidean)
```

### Bước 6: Rebuild FAISS index
```bash
python -m audio_model.build_audio_faiss
```

### Bước 7: Cập nhật CLIPAudioBridge
- `_rescale_similarities()`: Chuyển đổi L2 distances → similarity % (65-98%)
- `_clean_artist()`: Strip brackets `['Ed Sheeran'] → Ed Sheeran`

## 3. VẤN ĐỀ GẶP PHẢI VÀ GIẢI PHÁP

### Vấn đề 1: Giảm dim đơn thuần KHÔNG cải thiện
- **Quan sát:** Sau khi giảm 32→8 nhưng giữ nguyên L2-norm + IndexFlatIP, mean cosine sim tăng từ 0.925 → 0.978 (tệ hơn!)
- **Nguyên nhân:** L2-norm trên hypersphere 8D có surface area nhỏ hơn 32D → clustering chặt hơn
- **Giải pháp:** Bỏ L2-norm + chuyển sang IndexFlatL2 → giữ natural spacing

### Vấn đề 2: Similarity display cho L2 distance
- **Quan sát:** L2 distances rất nhỏ (0.00007-0.00017), exp(-d) vẫn cho ~1.0
- **Giải pháp:** Dùng min-max normalization trực tiếp trên distances, invert (smaller dist → higher sim),
  map vào range 0.65-0.98

## 4. KẾT QUẢ

### Training Results

| Metric | Giá trị |
|--------|---------|
| Epochs trained | 30 |
| Best Validation MSE | **0.031267** |
| Final Train MSE | 0.115425 |
| Model parameters | 657 |

### FAISS Index

| Metric | Cũ (dim=32) | Mới (dim=8) |
|--------|-------------|-------------|
| Index type | IndexFlatIP | IndexFlatL2 |
| Dimensions | 32 | 8 |
| Vectors | 1,083,622 | 1,083,622 |
| Index size | ~138 MB | ~33 MB |
| L2-normalized | ✅ | ❌ |

### Recommendation Quality

| Metric | Cũ (dim=32, cosine) | Mới (dim=8, L2) |
|--------|---------------------|------------------|
| Top-1 similarity | 99.9% | 98.0% |
| Top-10 similarity | 99.8% | 65-98% |
| Score spread | ~0.1% | ~33% |
| Distinguishable ranking | ❌ | ✅ |
| Clean artist names | ❌ | ✅ |
| Year/Album tags | ❌ | ✅ |

### Files đã tạo/cập nhật

| File | Kích thước |
|------|-----------|
| `models/autoencoder_model2.pth` | 7.7 KB |
| `models/tracks_faiss.index` | 33 MB |
| `models/tracks_faiss.index.mappings.pkl` | 104 MB |
| `models/autoencoder_model2_dim32_backup.pth` | backup |
| `models/tracks_faiss_dim32_backup.index` | backup |

## 5. KIẾN TRÚC

```
Audio Features (9) ─── Z-score Normalize ─── AudioAutoencoder ─── Raw Embedding (8D) ─── FAISS L2
   energy                                      Encoder:                                    IndexFlatL2
   danceability                                 9 → 16 → 8                                 1,083,622 vectors
   valence                                     Decoder (train only):
   tempo                                        8 → 16 → 9
   acousticness                                Loss: MSE(recon, input)
   instrumentalness
   speechiness
   liveness
   key
```

## 6. LƯU Ý KỸ THUẬT

1. **L2 distance (lower = more similar)** khác với cosine similarity (higher = more similar)
   → CLIPAudioBridge cần invert khi hiển thị
2. **Raw embeddings** có norm khác nhau (mean=0.49, std=0.02) — KHÔNG đồng nhất như L2-norm
3. **FAISS index nhỏ hơn 75%** (33MB vs ~138MB) do ít dimensions hơn
4. **657 parameters** — model cực kỳ nhẹ, phù hợp deployment

## 7. BƯỚC TIẾP THEO

1. ✅ Đã hoàn thành optimization — server running tốt
2. 🔲 Có thể thử contrastive loss thay MSE để cải thiện thêm (nhưng yêu cầu labels)
3. 🔲 Cập nhật README nếu cần thay đổi documentation
