# 🎵 Giải thích Module `audio_model/` — Model 2: Audio Autoencoder

> Module này chứa toàn bộ code của **Model 2** — mô hình học đặc trưng âm thanh thuần túy (không dùng lời bài hát).

---

## Tổng quan các file

| File | Vai trò |
|---|---|
| `config2.py` | Cấu hình tập trung (đường dẫn, siêu tham số) |
| `model2.py` | Định nghĩa kiến trúc mạng Autoencoder |
| `train_audio_model.py` | Vòng lặp huấn luyện mô hình |
| `build_audio_faiss.py` | Xây dựng FAISS index từ mô hình đã train |
| `clip_audio_bridge.py` | Cầu nối từ ảnh → nhãn cảm xúc → gợi ý nhạc |

---

## 📄 File 1: `config2.py` — Cấu hình tập trung

Tất cả siêu tham số và đường dẫn được đặt ở một chỗ duy nhất.

```python
@dataclass
class AudioOnlyConfig:
    # Thư mục dữ liệu
    data_dir = "data/processed/tracks"

    # File input (output của prepare_tracks_data.py)
    train_features_path = "data/processed/tracks/features_train.npy"
    train_meta_path     = "data/processed/tracks/meta_train.parquet"
    stats_path          = "models/tracks_stats.json"   # mean/std để normalize

    # File output sau khi train
    model_path          = "models/autoencoder_model2.pth"
    faiss_index_path    = "models/tracks_faiss.index"
    faiss_mappings_path = "models/tracks_faiss.index.mappings.pkl"

    # 9 đặc trưng âm thanh (không có emotion)
    audio_features = ['energy', 'danceability', 'valence', 'tempo',
                      'acousticness', 'instrumentalness', 'speechiness', 'liveness', 'key']

    # Kiến trúc mạng: 9 → 16 → 32 (bottleneck) → 16 → 9
    input_dim  = 9
    output_dim = 32   # Chiều của không gian latent

    # Siêu tham số huấn luyện
    batch_size = 2048
    epochs     = 30
    learning_rate = 1e-3
    early_stopping_patience = 5

CONFIG = AudioOnlyConfig()   # Import từ các file khác: from audio_model.config2 import CONFIG
```

---

## 📄 File 2: `model2.py` — Kiến trúc mạng Autoencoder

### Autoencoder là gì?

Hãy tưởng tượng bạn cần nén một bức ảnh 4K xuống thumbnail, rồi phục hồi lại. Phần nén gọi là **Encoder**, phần phục hồi gọi là **Decoder**. Vùng giữa (thumbnail) gọi là **Latent space** — đây là phần ta dùng để so sánh các bài hát.

```
Input (9 đặc trưng)
        ↓
   ENCODER: 9 → 16 → 32    ← "Nén" thông tin
        ↓
   Latent vector (32 chiều) ← Dùng cho FAISS search
        ↓
   DECODER: 32 → 16 → 9    ← "Giải nén" để kiểm tra
        ↓
Reconstruction (9 đặc trưng) ← So sánh với input để tính Loss
```

```python
class AudioAutoencoder(nn.Module):
    def __init__(self, input_dim=9, latent_dim=32, dropout=0.1):
        super().__init__()

        # ENCODER: thu nhỏ từ 9 → 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),  # Lớp tuyến tính 9 → 16
            nn.LayerNorm(16),           # Chuẩn hóa để ổn định training
            nn.GELU(),                  # Hàm kích hoạt (phi tuyến)
            nn.Dropout(dropout),        # Ngẫu nhiên tắt neuron để tránh overfit

            nn.Linear(16, latent_dim), # Lớp tuyến tính 16 → 32 (bottleneck)
        )

        # DECODER: phục hồi từ 32 → 9
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(16, input_dim),  # Phục hồi về 9 chiều ban đầu
        )

    def forward(self, x):
        """Dùng khi TRAINING — trả về cả embedding lẫn reconstruction."""
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        emb    = F.normalize(latent, p=2, dim=-1)  # L2-normalize → dùng cho FAISS
        return {'embedding': emb, 'reconstruction': recon}

    def encode(self, x):
        """Dùng khi INFERENCE/FAISS — chỉ cần latent vector, không cần decode."""
        latent = self.encoder(x)
        return F.normalize(latent, p=2, dim=-1)  # shape: [batch, 32]
```

> **Tại sao L2-normalize?** Sau khi normalize, tất cả vector đều có độ dài = 1 (nằm trên mặt cầu đơn vị). Lúc này inner product = cosine similarity. FAISS `IndexFlatIP` sẽ đo độ tương đồng theo cosine thay vì khoảng cách thô.

---

## 📄 File 3: `train_audio_model.py` — Vòng lặp huấn luyện

### Cách mô hình học

Không có nhãn đúng/sai → Mô hình tự học bằng cách cố gắng **tái tạo lại input** của chính nó (Unsupervised Learning).

```python
def load_features(npy_path, stats_path):
    """Tải features .npy và Z-score normalize theo stats đã tính từ train set."""
    features = np.load(npy_path).astype(np.float32)  # shape: (N, 9)

    with open(stats_path) as f:
        stats = json.load(f)

    means = np.array(stats["mean"], dtype=np.float32)
    stds  = np.array(stats["std"],  dtype=np.float32)

    # Z-score: (x - mean) / std  →  phân phối chuẩn, trung bình ≈ 0
    features = (features - means) / (stds + 1e-8)
    return torch.from_numpy(features)


def train_epoch(loader, model, optimizer, criterion, device):
    model.train()
    for (x,) in loader:
        x = x.to(device)
        optimizer.zero_grad()

        out  = model(x)
        loss = criterion(out["reconstruction"], x)   # MSE(tái tạo, gốc)
        # Loss thấp → mô hình tái tạo tốt → latent space có nghĩa

        loss.backward()    # Tính gradient
        optimizer.step()   # Cập nhật trọng số


def main():
    # Khởi tạo mô hình
    model = AudioAutoencoder(input_dim=9, latent_dim=32, dropout=0.1)

    # Optimizer: AdamW (tốt hơn Adam vì có weight decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Scheduler: giảm learning rate theo dạng cosine → ổn định hội tụ
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    criterion = nn.MSELoss()  # Đo sai lệch bình phương trung bình

    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, 31):
        train_loss = train_epoch(train_loader, model, optimizer, criterion, device)
        val_loss   = eval_epoch(val_loader, model, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())  # Lưu checkpoint tốt nhất
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:   # Early stopping: dừng nếu 5 epoch không cải thiện
                break

    # Lưu mô hình tốt nhất
    torch.save(best_state, "models/autoencoder_model2.pth")
```

---

## 📄 File 4: `build_audio_faiss.py` — Xây dựng FAISS Index

FAISS = **Facebook AI Similarity Search** — thư viện tìm kiếm vector gần nhất cực nhanh.

### Quy trình

```
1.2M bài hát (9 đặc trưng mỗi bài)
        ↓  Z-score normalize
        ↓  AudioAutoencoder.encode()
1.2M vectors (32 chiều mỗi vector)
        ↓  faiss.normalize_L2()
        ↓  faiss.IndexFlatIP.add()
FAISS Index (tra cứu cosine similarity)
```

```python
def build_index(model_path, train_features_path, train_meta_path, stats_path,
                faiss_out, mappings_out, latent_dim=32, batch_size=4096):

    # 1. Tải mô hình đã train
    model = AudioAutoencoder(input_dim=9, latent_dim=32, dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 2. Tải và Z-score normalize toàn bộ 1.2M bài
    features = load_and_normalize_npy(train_features_path, stats_path)

    # 3. Encode từng batch qua Autoencoder → latent vectors
    all_embeddings = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=4096):
            emb = model.encode(batch)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)  # shape: (1.2M, 32)

    # 4. L2-normalize → IndexFlatIP sẽ đo cosine similarity
    faiss.normalize_L2(embeddings)

    # 5. Xây dựng FAISS index
    index = faiss.IndexFlatIP(latent_dim)  # IP = Inner Product
    index.add(embeddings)                  # Thêm 1.2M vectors vào index

    # 6. Lưu index và mappings (index_position → thông tin bài hát)
    faiss.write_index(index, "models/tracks_faiss.index")

    meta_df  = pd.read_parquet(train_meta_path)
    mappings = [{"song": row["song_name"], "artist": row["artist"]} for _, row in meta_df.iterrows()]
    pickle.dump(mappings, open("models/tracks_faiss.index.mappings.pkl", "wb"))
```

---

## 📄 File 5: `clip_audio_bridge.py` — Cầu nối Ảnh → Nhạc

### Luồng hoạt động đầy đủ

```
[Người dùng upload ảnh]
        ↓
[CLIP model phân loại] → nhãn cảm xúc (vd: "Happy")
        ↓
[AUDIO_PROFILES bảng tra cứu]
        ↓  lookup "Happy"
[9 đặc trưng âm thanh phù hợp với Happy]
        ↓  Z-score normalize
        ↓  AudioAutoencoder.encode()
[Query vector 32 chiều]
        ↓  FAISS IndexFlatIP.search(top_k=10)
[10 bài hát gần nhất]
```

```python
# Bảng Audio Profiles — ánh xạ tâm trạng → 9 đặc trưng âm thanh
# Cột: [energy, danceability, valence, tempo, acousticness, instrumentalness, speechiness, liveness, key]
AUDIO_PROFILES = {
    "Happy":     [0.75, 0.72, 0.85, 120.0, 0.15, 0.05, 0.10, 0.15, 0.45],
    "Sad":       [0.30, 0.38, 0.18,  82.0, 0.62, 0.10, 0.05, 0.08, 0.45],
    "Energetic": [0.92, 0.82, 0.75, 142.0, 0.05, 0.03, 0.08, 0.22, 0.45],
    "Calm":      [0.22, 0.42, 0.55,  88.0, 0.72, 0.22, 0.04, 0.10, 0.45],
    "Angry":     [0.88, 0.55, 0.15, 135.0, 0.08, 0.05, 0.15, 0.28, 0.45],
    "Party":     [0.87, 0.88, 0.80, 126.0, 0.08, 0.03, 0.10, 0.30, 0.45],
    "Workout":   [0.92, 0.78, 0.70, 142.0, 0.05, 0.03, 0.08, 0.18, 0.45],
    "Study":     [0.28, 0.38, 0.50,  95.0, 0.68, 0.42, 0.04, 0.10, 0.45],
    "Relax":     [0.18, 0.42, 0.60,  84.0, 0.78, 0.28, 0.03, 0.10, 0.45],
    "Driving":   [0.72, 0.65, 0.65, 116.0, 0.18, 0.05, 0.06, 0.15, 0.45],
}


class CLIPAudioBridge:
    def __init__(self):
        # Tải stats để Z-score normalize query
        stats = json.load(open(CONFIG.stats_path))
        self._means = np.array(stats["mean"], dtype=np.float32)
        self._stds  = np.array(stats["std"],  dtype=np.float32)

        # Tải Autoencoder đã train
        self.model = AudioAutoencoder(input_dim=9, latent_dim=32, dropout=0.0)
        self.model.load_state_dict(torch.load(CONFIG.model_path))
        self.model.eval()

        # Tải FAISS index (1.2M vectors)
        self.faiss_index = faiss.read_index(CONFIG.faiss_index_path)

        # Tải bảng ánh xạ index → thông tin bài hát
        self.mappings = pickle.load(open(CONFIG.faiss_mappings_path, "rb"))

    def _normalize(self, raw):
        """Z-score normalize một profile âm thanh."""
        arr = np.array(raw, dtype=np.float32)
        return (arr - self._means) / (self._stds + 1e-8)

    def _encode(self, norm_features):
        """Encode features → 32D L2-unit embedding."""
        tensor = torch.tensor(norm_features).unsqueeze(0)  # thêm batch dim
        with torch.no_grad():
            emb = self.model.encode(tensor)
        emb_np = emb.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(emb_np)   # Unit-normalize
        return emb_np

    def recommend_from_label(self, label, top_k=10):
        """Nhận nhãn cảm xúc → trả về top-k gợi ý."""
        raw_profile   = AUDIO_PROFILES.get(label.capitalize(), AUDIO_PROFILES["Calm"])
        norm_features = self._normalize(raw_profile)
        query_emb     = self._encode(norm_features)   # shape: (1, 32)

        distances, indices = self.faiss_index.search(query_emb, top_k)

        # Chuyển cosine similarity thô (≈0.999 cho tất cả) sang thang tương đối 50-95%
        scores = self._relative_scores([float(d) for d in distances[0]])

        return [
            {"song": self.mappings[i]["song"], "artist": self.mappings[i]["artist"],
             "similarity": s}
            for s, i in zip(scores, indices[0]) if i >= 0
        ]

    @staticmethod
    def _relative_scores(distances, top_score=0.95, bottom_score=0.50):
        """
        Tại sao cần hàm này?
        Autoencoder train để TÁI TẠO, không phải PHÂN BIỆT.
        → Latent space "co cụm": mọi bài hát đều có cosine similarity ≈ 0.999
        → Không thể hiển thị 99.9% cho tất cả → cần re-scale về khoảng có ý nghĩa
        """
        max_d, min_d = max(distances), min(distances)
        span = max_d - min_d
        if span < 1e-9:
            return [(top_score + bottom_score) / 2] * len(distances)
        return [
            round(bottom_score + (d - min_d) / span * (top_score - bottom_score), 4)
            for d in distances
        ]
```

---

## 🗺️ Sơ đồ luồng dữ liệu của toàn Module

```
prepare_tracks_data.py
  └─ features_train.npy  ──────────────────────────────┐
  └─ meta_train.parquet  ──────────────────────────┐   │
  └─ tracks_stats.json  ───────────────────────┐   │   │
                                               │   │   │
train_audio_model.py                           │   │   │
  ├─ load_features() ←──────────────── stats ──┘   │   │
  │       ↓ normalize                              │   │
  ├─ AudioAutoencoder.forward() ←─── features ─────│───┘
  │       ↓ MSE Loss                               │
  └─ best_model → autoencoder_model2.pth           │
                                                   │
build_audio_faiss.py                               │
  ├─ load model ← autoencoder_model2.pth           │
  ├─ load_and_normalize() ←─── stats + features ───┘
  ├─ model.encode() → 1.2M vectors (32D)
  ├─ faiss.normalize_L2()
  ├─ IndexFlatIP.add()
  ├─ faiss.write_index() → tracks_faiss.index
  └─ pickle.dump()     → tracks_faiss.index.mappings.pkl

clip_audio_bridge.py (runtime - web app gọi)
  ├─ Tải model + FAISS + mappings + stats
  ├─ recommend_from_label("Happy")
  │    └─ AUDIO_PROFILES["Happy"] → normalize → encode → FAISS search
  └─ recommend_from_song("Shape of You")
       └─ tìm bài trong mappings → lấy vector từ FAISS → search
```
