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

Tất cả siêu tham số và đường dẫn được đặt ở **một chỗ duy nhất** (Single Source of Truth). Mọi file khác (`train`, `build`, `bridge`) đều `from audio_model.config2 import CONFIG` — không hardcode đường dẫn rải rác.

```python
@dataclass
class AudioOnlyConfig:
    # ── Dữ liệu đầu vào ───────────────────────────────────────────────────────
    data_dir = "data/processed/tracks"

    # features_train.npy  : ma trận numpy shape (N, 9) — N bài hát, 9 đặc trưng âm thanh
    # meta_train.parquet  : bảng thông tin bài hát (song_name, artist, genre...)
    # tracks_stats.json   : mean & std của từng feature, tính trên TẬP TRAIN
    #                       Quan trọng: phải dùng stats của train set để normalize cả val/test
    #                       tránh "data leakage" (rò rỉ thông tin từ val vào train)
    train_features_path = "data/processed/tracks/features_train.npy"
    train_meta_path     = "data/processed/tracks/meta_train.parquet"
    stats_path          = "models/tracks_stats.json"

    # ── File output sau khi train ─────────────────────────────────────────────
    # autoencoder_model2.pth            : trọng số mạng neural đã học
    # tracks_faiss.index                : FAISS index chứa 1.2M vector 32D
    # tracks_faiss.index.mappings.pkl   : danh sách dict [{song, artist}]
    #   → vị trí i trong mappings tương ứng với vector thứ i trong FAISS index
    model_path          = "models/autoencoder_model2.pth"
    faiss_index_path    = "models/tracks_faiss.index"
    faiss_mappings_path = "models/tracks_faiss.index.mappings.pkl"

    # ── 9 đặc trưng âm thanh (Spotify Audio Features) ─────────────────────────
    # Không dùng text/lời bài hát — chỉ dùng đặc trưng vật lý/âm nhạc.
    # energy          : cường độ và hoạt động (0→1). Nhạc metal ≈ 0.9, nhạc thư giãn ≈ 0.2
    # danceability    : phù hợp để nhảy (0→1), tính từ beat, tempo, nhịp điệu
    # valence         : độ "vui" (0→1). Buồn ≈ 0.1, vui ≈ 0.9
    # tempo           : nhịp độ (BPM). Ballad ≈ 70, EDM ≈ 140
    # acousticness    : tỷ lệ nhạc cụ acoustic (0→1). Guitar acoustic ≈ 0.9
    # instrumentalness: không có giọng hát (0→1). Nhạc không lời ≈ 0.9
    # speechiness     : nhiều lời nói/rap (0→1). Podcast ≈ 0.9, nhạc thường ≈ 0.05
    # liveness        : khả năng thu âm trực tiếp (live) (0→1)
    # key             : điệu (0-11: C, C#, D... B). Được normalize thành 0→1
    audio_features = ['energy', 'danceability', 'valence', 'tempo',
                      'acousticness', 'instrumentalness', 'speechiness', 'liveness', 'key']

    # ── Kiến trúc mạng ────────────────────────────────────────────────────────
    # input_dim=9   : đúng bằng số đặc trưng âm thanh
    # output_dim=32 : kích thước latent space — đủ lớn để phân biệt, đủ nhỏ để FAISS nhanh
    #   Tại sao 32? Thực nghiệm: 16D mất thông tin, 64D không cải thiện đáng kể
    input_dim  = 9
    output_dim = 32

    # ── Siêu tham số huấn luyện ───────────────────────────────────────────────
    # batch_size=2048 : lớn → gradient ổn định hơn, tận dụng GPU tốt hơn
    # epochs=30       : giới hạn trên; early stopping thường dừng sớm hơn
    # learning_rate=1e-3 : mặc định tốt cho AdamW với dataset lớn
    # early_stopping_patience=5 : dừng nếu val_loss không cải thiện 5 epoch liên tiếp
    batch_size = 2048
    epochs     = 30
    learning_rate = 1e-3
    weight_decay  = 1e-4  # L2 regularization trong AdamW
    dropout       = 0.1   # xác suất tắt neuron khi training
    early_stopping_patience = 5

CONFIG = AudioOnlyConfig()   # Singleton — import từ bất kỳ file nào trong module
```

---

## 📄 File 2: `model2.py` — Kiến trúc mạng Autoencoder

### Autoencoder là gì?

Hãy tưởng tượng bạn cần nén một bức ảnh 4K xuống thumbnail, rồi phục hồi lại. Phần nén gọi là **Encoder**, phần phục hồi gọi là **Decoder**. Vùng giữa (thumbnail) gọi là **Latent space** — đây là phần ta dùng để so sánh các bài hát.

**Tại sao dùng Autoencoder thay vì supervised model?**
- Không có nhãn "bài hát này giống bài hát kia" trong dataset
- Autoencoder học unsupervised: tự tổ chức không gian latent để bài hát có âm thanh tương tự sẽ có vector gần nhau
- Decoder chỉ là "phương tiện" để ép Encoder học tốt — sau training, Decoder bị bỏ đi

```
Input (9 đặc trưng âm thanh của 1 bài hát)
        ↓
   ENCODER: 9 → 16 → 32    ← "Nén" — buộc mạng học biểu diễn cô đọng
        ↓
   Latent vector (32 chiều) ← "DNA âm nhạc" của bài hát
        ↓                      Dùng cho FAISS search khi inference
   DECODER: 32 → 16 → 9    ← "Giải nén" — chỉ dùng khi training
        ↓
Reconstruction (9 đặc trưng) ← So sánh với input gốc để tính Loss
                               Loss thấp = Encoder học tốt
```

```python
class AudioAutoencoder(nn.Module):
    def __init__(self, input_dim=9, latent_dim=32, dropout=0.1):
        super().__init__()  # Kế thừa nn.Module — bắt buộc trong PyTorch

        # ── ENCODER: 9 → 16 → 32 ──────────────────────────────────────────
        self.encoder = nn.Sequential(
            # Bước 1: Linear(9→16)
            # Ma trận trọng số W shape (9,16) + bias (16,)
            # Phép tính: output = x @ W.T + bias
            nn.Linear(input_dim, 16),

            # Bước 2: LayerNorm(16)
            # Normalize TRÊN TỪNG SAMPLE (không phải batch)
            # Công thức: (x - mean(x)) / std(x) * gamma + beta
            # Giúp gradient không bị vanish/explode, training ổn định hơn BatchNorm
            # với batch nhỏ hoặc sequence data
            nn.LayerNorm(16),

            # Bước 3: GELU (Gaussian Error Linear Unit)
            # GELU(x) = x * Φ(x)  với Φ là CDF của phân phối chuẩn
            # Gần giống ReLU nhưng mượt hơn tại 0 → gradient không bị chặn cứng
            # Được dùng trong BERT, GPT — hiệu quả hơn ReLU cho nhiều tác vụ
            nn.GELU(),

            # Bước 4: Dropout(0.1)
            # Khi training: ngẫu nhiên tắt 10% neuron → mỗi lần forward khác nhau
            # → mạng không thể memorize, buộc học đặc trưng tổng quát hơn
            # Khi inference (model.eval()): Dropout tự động bị tắt
            nn.Dropout(dropout),

            # Bước 5: Linear(16→32) — BOTTLENECK
            # Đây là điểm nút: ép 9 features qua cổ chai 16 chiều rồi mở ra 32 chiều
            # 32 > 9 nhưng thông tin đã được tái cấu trúc theo cách có ý nghĩa hơn
            # (không phải raw features nữa mà là learned representation)
            nn.Linear(16, latent_dim),
        )

        # ── DECODER: 32 → 16 → 9 (đối xứng với Encoder) ──────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),  # Mở rộng từ latent về intermediate
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, input_dim),   # Phục hồi về 9 chiều ban đầu
            # KHÔNG có activation ở cuối Decoder vì output là giá trị liên tục
            # (không phải xác suất), MSELoss không yêu cầu sigmoid/softmax
        )

    def forward(self, x: torch.Tensor):
        """
        Gọi khi TRAINING.
        x shape: (batch_size, 9)
        
        Trả về dict với 2 key:
          - 'embedding'      : L2-normalized latent, shape (batch, 32)
          - 'reconstruction' : output của decoder, shape (batch, 9)
        
        Loss được tính bên ngoài:
          loss = MSELoss(output['reconstruction'], x)
        """
        latent = self.encoder(x)                   # (batch, 9) → (batch, 32)
        recon  = self.decoder(latent)              # (batch, 32) → (batch, 9)
        # L2-normalize: chia mỗi vector cho độ dài Euclidean của nó
        # Sau normalize: ||emb||₂ = 1 (nằm trên mặt cầu đơn vị)
        # Mục đích: để inner product (dot product) = cosine similarity
        emb    = F.normalize(latent, p=2, dim=-1)  # p=2: L2 norm, dim=-1: normalize theo chiều cuối
        return {'embedding': emb, 'reconstruction': recon}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gọi khi INFERENCE / xây dựng FAISS index.
        Không cần decoder — chỉ lấy latent vector.
        
        x shape: (batch_size, 9)  hoặc  (1, 9) cho single song
        Returns: (batch_size, 32) — L2-normalized embeddings
        """
        latent = self.encoder(x)
        return F.normalize(latent, p=2, dim=-1)
```

> **Tại sao L2-normalize?**
>
> Giả sử vector A = [3, 4] → ||A|| = 5 → sau normalize: [0.6, 0.8]
>
> Khi tất cả vector đều có độ dài = 1:
> - `A · B = |A||B|cos(θ) = 1×1×cos(θ) = cos(θ)`
> - Inner product (dot product) đúng bằng cosine similarity
> - FAISS `IndexFlatIP` (Inner Product) trở thành cosine similarity search
> - Tìm được bài hát "cùng phong cách" thay vì "gần về giá trị số tuyệt đối"

---

## 📄 File 3: `train_audio_model.py` — Vòng lặp huấn luyện

### Cách mô hình học

Không có nhãn đúng/sai → Mô hình tự học bằng cách cố gắng **tái tạo lại input** của chính nó (Unsupervised / Self-supervised Learning). Ý tưởng: nếu Encoder có thể nén 9 features xuống 32 chiều mà Decoder vẫn phục hồi lại được, thì 32 chiều đó **phải mang thông tin âm nhạc thực sự**.

```python
def load_features(npy_path, stats_path):
    """
    Bước 1: Tải dữ liệu và chuẩn hóa (Z-score Normalization).

    Tại sao phải normalize?
    - tempo có giá trị 60-200, energy có giá trị 0-1
    - Nếu không normalize: gradient của tempo sẽ rất lớn, lấn át energy
    - Sau Z-score: mọi feature đều có mean≈0, std≈1 → mạng học đồng đều

    Tại sao dùng stats của TẬP TRAIN?
    - Nếu dùng stats của toàn bộ data: val set "rò rỉ" thông tin vào train
    - Luôn fit (tính mean/std) trên train, transform cả train + val + test
    """
    features = np.load(npy_path).astype(np.float32)  # shape: (N, 9)

    with open(stats_path) as f:
        stats = json.load(f)  # {"mean": [9 values], "std": [9 values]}

    means = np.array(stats["mean"], dtype=np.float32)  # shape: (9,)
    stds  = np.array(stats["std"],  dtype=np.float32)  # shape: (9,)

    # Z-score: đưa về phân phối chuẩn N(0,1)
    # + 1e-8 để tránh chia cho 0 nếu std của feature nào đó = 0
    features = (features - means) / (stds + 1e-8)
    return torch.from_numpy(features)  # PyTorch tensor để đưa vào DataLoader


def train_epoch(loader, model, optimizer, criterion, device):
    """
    Bước 2a: Một epoch huấn luyện — duyệt qua TOÀN BỘ training set 1 lần.
    """
    model.train()  # BẬT training mode: Dropout hoạt động, BatchNorm dùng batch stats
    total_loss = 0.0
    total_n = 0

    for (x,) in loader:  # x shape: (batch_size, 9)
        x = x.to(device)  # chuyển lên GPU nếu có

        # --- Forward pass ---
        optimizer.zero_grad()          # Xóa gradient cũ (PyTorch tích lũy gradient)
        out  = model(x)                # Gọi forward(): encode + decode
        loss = criterion(out["reconstruction"], x)
        # MSELoss = mean( (recon_i - x_i)^2 ) qua tất cả features và samples
        # Loss thấp → recon ≈ input → Encoder học được biểu diễn tốt

        # --- Backward pass ---
        loss.backward()    # Lan truyền ngược: tính d(loss)/d(weight) cho mọi tham số
        optimizer.step()   # Cập nhật: weight -= lr * gradient (AdamW có momentum)

        # Tích lũy để tính trung bình sau
        total_loss += loss.item() * x.size(0)  # .item() lấy scalar Python từ tensor
        total_n += x.size(0)

    return total_loss / total_n  # Trung bình loss PER SAMPLE


@torch.no_grad()  # Tắt tính gradient → tiết kiệm bộ nhớ, tăng tốc
def eval_epoch(loader, model, criterion, device):
    """
    Bước 2b: Đánh giá trên validation set — KHÔNG cập nhật trọng số.
    @torch.no_grad(): không lưu computation graph → ít VRAM hơn ~40%
    """
    model.eval()  # TẮT training mode: Dropout bị tắt, BatchNorm dùng running stats
    total_loss = 0.0
    total_n = 0
    for (x,) in loader:
        x = x.to(device)
        out  = model(x)
        loss = criterion(out["reconstruction"], x)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / total_n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cuda: GPU NVIDIA (training nhanh hơn ~10-50x)
    # cpu:  fallback nếu không có GPU

    # --- Khởi tạo DataLoader ---
    # DataLoader: chia dataset thành mini-batch, shuffle sau mỗi epoch
    # num_workers=0: load data trên main thread (an toàn trên Windows)
    # pin_memory=True: copy data vào pinned memory → transfer lên GPU nhanh hơn
    train_loader = DataLoader(TensorDataset(train_feats), batch_size=2048,
                              shuffle=True, num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(TensorDataset(val_feats),   batch_size=4096,
                              shuffle=False, num_workers=0)

    # --- Khởi tạo Model ---
    model = AudioAutoencoder(input_dim=9, latent_dim=32, dropout=0.1).to(device)
    # .to(device): chuyển tất cả trọng số lên GPU (nếu có)

    # --- Optimizer: AdamW ---
    # Adam = Adaptive Moment Estimation: điều chỉnh lr riêng cho từng tham số
    # AdamW = Adam + Weight Decay ĐÚNG CÁCH (Adam gốc có bug với L2 regularization)
    # weight_decay=1e-4: phạt trọng số lớn → tránh overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # --- Scheduler: CosineAnnealingLR ---
    # lr bắt đầu ở 1e-3, giảm dần theo hình cosine, kết thúc ≈ 0 sau T_max epoch
    # Lý do: lr lớn ban đầu để học nhanh, lr nhỏ cuối để tinh chỉnh (fine-tune)
    # Tốt hơn StepLR (giảm đột ngột) về độ ổn định của loss
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # --- Loss Function: MSELoss ---
    # MSE = (1/N) * Σ(recon_i - x_i)^2
    # Phù hợp cho regression/reconstruction vì output là giá trị liên tục
    # (Không dùng CrossEntropy vì đó là cho classification)
    criterion = nn.MSELoss()

    # --- Training Loop với Early Stopping ---
    best_val_loss    = float("inf")  # Khởi tạo ở vô cực
    best_state       = None          # Lưu state_dict của model tốt nhất
    patience_counter = 0             # Đếm số epoch val_loss không cải thiện

    for epoch in range(1, 31):
        train_loss = train_epoch(train_loader, model, optimizer, criterion, device)
        val_loss   = eval_epoch(val_loader,   model, criterion, device)
        scheduler.step()  # Cập nhật lr SAU khi đã eval

        if val_loss < best_val_loss:
            # Model cải thiện → lưu checkpoint
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            # copy.deepcopy QUAN TRỌNG: tạo bản sao độc lập
            # Nếu chỉ gán best_state = model.state_dict(), sau khi tiếp tục train
            # best_state sẽ bị thay đổi theo (cùng reference)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                # Early Stopping: tránh overfitting khi model đã hội tụ
                # train_loss vẫn giảm nhưng val_loss tăng = dấu hiệu overfitting
                print(f"Early stopping sau {epoch} epochs")
                break

    # Khôi phục model tốt nhất (không phải model cuối cùng!)
    model.load_state_dict(best_state)
    torch.save(best_state, "models/autoencoder_model2.pth")
    # Chỉ lưu state_dict (dict trọng số), không lưu cả object model
    # → nhẹ hơn, linh hoạt hơn, có thể load vào bất kỳ kiến trúc tương thích nào
```

---

## 📄 File 4: `build_audio_faiss.py` — Xây dựng FAISS Index

FAISS = **Facebook AI Similarity Search** — thư viện tìm kiếm vector gần nhất cực nhanh, được dùng trong production tại Meta, Spotify, Netflix.

**Tại sao dùng FAISS thay vì vòng lặp thông thường?**
- Brute-force: so sánh query với 1.2M vectors → O(N) = ~1.2M phép tính cosine mỗi request
- FAISS IndexFlatIP: tối ưu BLAS, chạy song song, vẫn O(N) nhưng nhanh hơn 100-1000x
- Với index ANN (IndexIVFFlat): gần đúng nhưng O(√N), nhanh hơn nữa

### Quy trình xây dựng index (chạy 1 lần offline)

```
1.2M bài hát (features_train.npy, shape: 1.2M × 9)
        ↓  BƯỚC 1: Z-score normalize (dùng tracks_stats.json)
        ↓  BƯỚC 2: AudioAutoencoder.encode() theo batch 4096
1.2M vectors (shape: 1.2M × 32)  ← latent representations
        ↓  BƯỚC 3: faiss.normalize_L2()  → unit vectors
        ↓  BƯỚC 4: IndexFlatIP.add()     → nạp vào index
FAISS Index   → tracks_faiss.index          (file nhị phân)
Mappings List → tracks_faiss.index.mappings.pkl  (list dict)
```

```python
def build_index(model_path, train_features_path, train_meta_path, stats_path,
                faiss_out, mappings_out, latent_dim=32, batch_size=4096):

    # BƯỚC 1: Tải model đã train
    # dropout=0.0 khi inference: Dropout bị tắt HOÀN TOÀN (không phải xác suất nhỏ)
    model = AudioAutoencoder(input_dim=9, latent_dim=32, dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    # map_location="cpu": load về CPU dù model được train trên GPU
    model.eval()  # Tắt Dropout, chuyển sang inference mode

    # BƯỚC 2: Load và normalize dữ liệu
    # Dùng CÙNG stats_path với khi train → nhất quán
    features = load_and_normalize_npy(train_features_path, stats_path)
    # features shape: (N, 9), dtype: float32

    # BƯỚC 3: Encode toàn bộ dataset qua Autoencoder
    # Dùng batch để không OOM (Out of Memory) với 1.2M samples
    all_embeddings = []
    with torch.no_grad():  # Không cần gradient khi inference
        for (batch,) in DataLoader(
            TensorDataset(torch.from_numpy(features)),
            batch_size=4096  # Lớn hơn training batch (4096 vs 2048) vì không cần backprop
        ):
            emb = model.encode(batch)           # shape: (4096, 32)
            all_embeddings.append(emb.cpu().numpy())  # Chuyển về numpy để vstack

    # Ghép tất cả batch lại thành một ma trận lớn
    embeddings = np.vstack(all_embeddings).astype(np.float32)  # shape: (N, 32)

    # BƯỚC 4: L2-normalize cho FAISS
    # model.encode() đã normalize trong PyTorch, nhưng sau khi chuyển sang numpy
    # có thể có sai số floating point nhỏ → normalize lại cho chắc chắn
    # faiss.normalize_L2() sửa IN-PLACE: embeddings[i] /= ||embeddings[i]||
    faiss.normalize_L2(embeddings)

    # BƯỚC 5: Tạo và nạp FAISS index
    # IndexFlatIP = Flat (brute-force, exact search) + IP (Inner Product metric)
    # Sau L2-normalize: IP = cosine similarity
    # Flat = so sánh với TẤT CẢ vectors (100% chính xác, không gần đúng)
    index = faiss.IndexFlatIP(latent_dim)  # latent_dim=32: kích thước mỗi vector
    index.add(embeddings)                  # Nạp 1.2M vectors vào RAM
    # index.ntotal = 1.2M sau khi add xong

    # BƯỚC 6: Lưu index và bảng ánh xạ
    # tracks_faiss.index: file nhị phân chứa toàn bộ ma trận 1.2M×32
    faiss.write_index(index, faiss_out)

    # mappings: list có N phần tử, phần tử thứ i = thông tin bài hát thứ i trong index
    # Quan hệ: FAISS trả về indices[i] → mappings[indices[i]] = {song, artist}
    meta_df  = pd.read_parquet(train_meta_path)
    mappings = [
        {"song": row["song_name"], "artist": row["artist"],
         "genre": row.get("genre", ""), "emotion": row.get("emotion", "")}
        for _, row in meta_df.iterrows()
    ]
    # pickle: serialize Python object → binary file
    # Nhanh hơn JSON cho list lớn, giữ nguyên kiểu dữ liệu Python
    pickle.dump(mappings, open(mappings_out, "wb"))

    print(f"[OK] FAISS index: {index.ntotal} vectors, dim={latent_dim}")
    print(f"[OK] Mappings: {len(mappings)} entries")
```

**Thời gian và bộ nhớ ước tính (1.2M songs):**

| Bước | Thời gian | RAM |
|------|-----------|-----|
| Load .npy | ~2s | ~43MB (1.2M × 9 × 4B) |
| Encode (CPU) | ~3-5 phút | ~154MB (1.2M × 32 × 4B) |
| Encode (GPU) | ~20-30s | ~154MB VRAM |
| normalize_L2 | <1s | in-place |
| index.add() | ~5s | ~154MB thêm |
| write_index | ~2s | ~154MB file |

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
# Bảng Audio Profiles — ánh xạ tâm trạng/bối cảnh → 9 đặc trưng âm thanh
# Thứ tự cột: [energy, danceability, valence, tempo, acousticness,
#              instrumentalness, speechiness, liveness, key]
#
# Cách xây dựng bảng này:
# - Dựa trên định nghĩa của Spotify cho từng đặc trưng
# - Khảo sát âm nhạc học và tâm lý học âm nhạc (music psychology)
# - Điều chỉnh qua thực nghiệm
AUDIO_PROFILES = {
    #                  ener  danc  vale  tempo  acou  inst  spee  live  key
    "Happy":     [0.75, 0.72, 0.85, 120.0, 0.15, 0.05, 0.10, 0.15, 0.45],
    # Happy: tempo vừa phải (120 BPM = đi bộ nhanh), valence cao (0.85=vui),
    # acousticness thấp (không quá trầm lặng)

    "Sad":       [0.30, 0.38, 0.18,  82.0, 0.62, 0.10, 0.05, 0.08, 0.45],
    # Sad: energy thấp (0.30), valence rất thấp (0.18=buồn), tempo chậm (82 BPM),
    # acousticness cao (0.62) → thiên về guitar acoustic, piano

    "Energetic": [0.92, 0.82, 0.75, 142.0, 0.05, 0.03, 0.08, 0.22, 0.45],
    # Energetic: energy rất cao (0.92), tempo nhanh (142 BPM=chạy bộ),
    # danceability cao (0.82), acousticness gần 0 (nhạc điện tử)

    "Calm":      [0.22, 0.42, 0.55,  88.0, 0.72, 0.22, 0.04, 0.10, 0.45],
    # Calm: energy rất thấp (0.22), acousticness cao (0.72), valence trung bình
    # instrumentalness cao (0.22) → thiên về nhạc không lời ambient

    "Angry":     [0.88, 0.55, 0.15, 135.0, 0.08, 0.05, 0.15, 0.28, 0.45],
    # Angry: energy cao (0.88), valence rất thấp (0.15=âm tính), tempo nhanh
    # speechiness cao hơn (0.15) → heavy metal, rap dữ, rock cứng

    "Party":     [0.87, 0.88, 0.80, 126.0, 0.08, 0.03, 0.10, 0.30, 0.45],
    # Party: danceability rất cao (0.88), valence cao (0.80=vui),
    # liveness cao hơn (0.30) → cảm giác live show

    "Workout":   [0.92, 0.78, 0.70, 142.0, 0.05, 0.03, 0.08, 0.18, 0.45],
    # Workout ≈ Energetic nhưng valence cao hơn (0.70 vs 0.75) → tích cực, động lực

    "Study":     [0.28, 0.38, 0.50,  95.0, 0.68, 0.42, 0.04, 0.10, 0.45],
    # Study: energy và danceability thấp → không phân tâm,
    # instrumentalness cao (0.42) → nhạc không lời để tập trung

    "Relax":     [0.18, 0.42, 0.60,  84.0, 0.78, 0.28, 0.03, 0.10, 0.45],
    # Relax: energy rất thấp (0.18), acousticness cao (0.78=mộc mạc),
    # valence trung cao (0.60=dễ chịu) → jazz, acoustic pop nhẹ

    "Driving":   [0.72, 0.65, 0.65, 116.0, 0.18, 0.05, 0.06, 0.15, 0.45],
    # Driving: cân bằng giữa energy (0.72) và valence (0.65) → rocknhẹ, indie
    # tempo 116 BPM ≈ tốc độ xe hơi thành phố
}


class CLIPAudioBridge:
    """
    Cầu nối giữa CLIP (xử lý ảnh) và AudioAutoencoder (tìm nhạc).
    
    Khởi tạo khi server bắt đầu (ONCE) — load model + FAISS vào RAM.
    Các request sau đó dùng lại các object này → nhanh.
    """
    def __init__(self):
        # Tải stats để Z-score normalize query (CÙNG stats với khi train!)
        stats = json.load(open(CONFIG.stats_path))
        self._means = np.array(stats["mean"], dtype=np.float32)
        self._stds  = np.array(stats["std"],  dtype=np.float32)

        # Tải Autoencoder (chỉ dùng Encoder, Decoder không cần)
        # dropout=0.0: tắt Dropout khi inference
        self.model = AudioAutoencoder(input_dim=9, latent_dim=32, dropout=0.0)
        self.model.load_state_dict(torch.load(CONFIG.model_path, map_location="cpu"))
        self.model.eval()  # Bắt buộc: tắt training mode

        # Tải FAISS index (1.2M vectors 32D vào RAM ≈ 154MB)
        # Read lần đầu mất ~2s, sau đó search rất nhanh (~1ms)
        self.faiss_index = faiss.read_index(CONFIG.faiss_index_path)

        # Tải bảng ánh xạ: vị trí index → thông tin bài hát
        # self.mappings[i] = {"song": "...", "artist": "..."}
        self.mappings = pickle.load(open(CONFIG.faiss_mappings_path, "rb"))

    def _normalize(self, raw: list) -> np.ndarray:
        """Z-score normalize một audio profile.
        raw: list 9 giá trị thô (ví dụ [0.75, 0.72, 0.85, 120.0, ...])
        Returns: array 9 giá trị đã normalize, dùng cùng scale với training data
        """
        arr = np.array(raw, dtype=np.float32)
        return (arr - self._means) / (self._stds + 1e-8)

    def _encode(self, norm_features: np.ndarray) -> np.ndarray:
        """Encode một audio profile → query vector 32D cho FAISS.
        norm_features: array (9,) đã Z-score normalize
        Returns: array (1, 32) L2-unit — sẵn sàng cho FAISS.search()
        """
        # unsqueeze(0): thêm batch dimension: (9,) → (1, 9)
        # Model yêu cầu input có batch dim: (batch_size, input_dim)
        tensor = torch.tensor(norm_features).unsqueeze(0)
        with torch.no_grad():
            emb = self.model.encode(tensor)       # shape: (1, 32), L2-normalized
        emb_np = emb.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(emb_np)   # Đảm bảo unit-length sau khi chuyển numpy
        return emb_np

    def recommend_from_label(self, label: str, top_k: int = 10) -> list:
        """
        Nhập: nhãn cảm xúc/bối cảnh (“Happy”, “Party”...)
        Đầu ra: top-k bài hát phù hợp

        Lógic:
        1. Tra AUDIO_PROFILES[label] → 9 giá trị thô
        2. Z-score normalize → 9 giá trị chuẩn hóa
        3. Encode qua Autoencoder → query vector 32D
        4. FAISS.search(query, top_k) → top-k indices và distances
        5. Map indices → thông tin bài hát (từ self.mappings)
        6. Re-scale similarity về khoảng [50%, 95%] cho dễ hiểu
        """
        # .capitalize(): đảm bảo "happy" → "Happy" để khop key trong dict
        # Fallback về "Calm" nếu nhãn không tồn tại
        raw_profile   = AUDIO_PROFILES.get(label.capitalize(), AUDIO_PROFILES["Calm"])
        norm_features = self._normalize(raw_profile)   # (9,) normalized
        query_emb     = self._encode(norm_features)    # (1, 32) unit vector

        # FAISS search: tìm top_k vectors gần query_emb nhất
        # distances shape: (1, top_k) — cosine similarity scores
        # indices   shape: (1, top_k) — vị trí trong FAISS index
        distances, indices = self.faiss_index.search(query_emb, top_k)

        # Re-scale similarity
        scores = self._relative_scores([float(d) for d in distances[0]])

        return [
            {"song":       self.mappings[i]["song"],
             "artist":     self.mappings[i]["artist"],
             "genre":      self.mappings[i].get("genre", ""),
             "emotion":    self.mappings[i].get("emotion", ""),
             "similarity": s}
            for s, i in zip(scores, indices[0])
            if i >= 0  # FAISS trả -1 nếu index dưới ngưỡng, bỏ qua
        ]

    @staticmethod
    def _relative_scores(distances: list,
                         top_score: float = 0.95,
                         bottom_score: float = 0.50) -> list:
        """
        Tại sao cần hàm này?

        Vấn đề: Autoencoder train để TÁI TẠO, không phải PHÂN BIỆT.
        → Latent space “co cụm”: mọi bài hát đều có cosine similarity ≈ 0.999 với nhau
        → Không thể hiển thị 99.9% cho tất cả → cần re-scale về khoảng có ý nghĩa

        Giải pháp: Min-Max normalization trên top-k kết quả
        → bài hát gần nhất = 95%, xa nhất (trong top-k) = 50%
        → Thử tự tương đối được giữ nguyên, chỉ scale thì thay đổi

        Công thức:
          score_i = bottom + (d_i - d_min) / (d_max - d_min) * (top - bottom)
        """
        max_d, min_d = max(distances), min(distances)
        span = max_d - min_d
        if span < 1e-9:  # Tất cả bằng nhau (edge case) → trả giá trị trung bình
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
