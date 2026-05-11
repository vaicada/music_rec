# Huấn luyện Model 2 — Audio Autoencoder

> **Dành cho thuyết trình / slide**
> Nội dung học thuật, súc tích, có thể copy trực tiếp vào slide.

---

## SLIDE 1 — Tổng quan

### Tiêu đề: Model 2: Học Không Giám Sát với Audio Autoencoder

**Vấn đề đặt ra:**

- Có 1,2 triệu bài hát với **9 đặc trưng âm thanh** mỗi bài
- **Không có nhãn** "bài hát A giống bài hát B"
- Làm thế nào để máy tính "hiểu" bài hát nào tương đồng với nhau?

**Giải pháp:**

- Sử dụng **Autoencoder** — mô hình tự học cấu trúc dữ liệu mà không cần nhãn ngoài
- Phương pháp: **Unsupervised Learning (Học không giám sát)**

---

## SLIDE 2 — 9 Đặc Trưng Âm Thanh (Input)

### Tiêu đề: Dữ liệu đầu vào — Spotify Audio Features

| Đặc trưng | Ý nghĩa | Khoảng giá trị |
|---|---|---|
| `energy` | Cường độ và hoạt động âm nhạc | 0 → 1 |
| `danceability` | Mức độ phù hợp để nhảy | 0 → 1 |
| `valence` | Độ "vui" / tích cực của bài hát | 0 → 1 |
| `tempo` | Nhịp độ (BPM) | 60 → 200 |
| `acousticness` | Tỷ lệ nhạc cụ acoustic | 0 → 1 |
| `instrumentalness` | Không có giọng hát | 0 → 1 |
| `speechiness` | Mức độ lời nói / rap | 0 → 1 |
| `liveness` | Thu âm trực tiếp (live) | 0 → 1 |
| `key` | Điệu nhạc (C, D, E...) | 0 → 11 |

> **Lưu ý:** Tất cả giá trị được **Z-score normalize** trước khi đưa vào mô hình:
> $$\hat{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

---

## SLIDE 3 — Kiến Trúc Autoencoder

### Tiêu đề: Kiến trúc mô hình — Encoder & Decoder

```
INPUT (9 đặc trưng)
        │
        ▼
┌───────────────────┐
│     ENCODER       │   9 → 16 → 32
│  Linear(9→16)     │
│  LayerNorm + GELU │   ← học biểu diễn cô đọng
│  Linear(16→32)    │
└───────────────────┘
        │
        ▼
  ┌──────────────┐
  │ Latent Space │   32 chiều — "DNA âm nhạc" của bài hát
  │   (32D)      │   ← vector này dùng cho FAISS search
  └──────────────┘
        │
        ▼
┌───────────────────┐
│     DECODER       │   32 → 16 → 9
│  Linear(32→16)    │
│  LayerNorm + GELU │   ← tái tạo lại đầu vào
│  Linear(16→9)     │
└───────────────────┘
        │
        ▼
OUTPUT (9 đặc trưng — tái tạo)
```

**Tại sao 32 chiều?**

- 16D: mất quá nhiều thông tin
- 64D: không cải thiện đáng kể, tốn tài nguyên hơn
- **32D: điểm cân bằng tối ưu** qua thực nghiệm

---

## SLIDE 4 — Logic Huấn Luyện (Điểm Mấu Chốt)

### Tiêu đề: Tại sao không cần nhãn? — Self-supervised Learning

**Ý tưởng cốt lõi:**

> Nếu mô hình có thể **nén 9 đặc trưng xuống 32 chiều** rồi **tái tạo lại chính xác 9 đặc trưng ban đầu**, thì 32 chiều đó **phải mang thông tin âm nhạc thực sự**.

**Hàm mất mát — biến phụ thuộc là chính đầu vào:**

$$\mathcal{L}(\theta, \phi) = \frac{1}{N} \sum_{i=1}^{N} \left\| g_\phi\bigl(f_\theta(\mathbf{x}_i)\bigr) - \mathbf{x}_i \right\|_2^2$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $\mathbf{x}_i$ | Vector 9 đặc trưng bài hát thứ $i$ (INPUT) |
| $f_\theta(\mathbf{x}_i)$ | Latent vector 32D — output của Encoder |
| $g_\phi(\cdot)$ | Output của Decoder (tái tạo) |
| $\mathcal{L}$ | MSE Loss — sai số tái tạo |

**→ Target $y$ = chính $\mathbf{x}$ (Input). Không cần nhãn ngoài.**

---

## SLIDE 5 — Quá Trình Huấn Luyện (Training Loop)

### Tiêu đề: Các bước huấn luyện

**Mỗi epoch thực hiện:**

```
1. Lấy batch 2.048 bài hát (đã normalize)
        ↓
2. Forward pass: Encoder → Latent → Decoder → Reconstruction
        ↓
3. Tính MSE Loss: so sánh Reconstruction với Input gốc
        ↓
4. Backward pass: tính gradient d(Loss)/d(weights)
        ↓
5. AdamW cập nhật trọng số: giảm Loss
        ↓
6. Lặp lại cho hết tập train
        ↓
7. Đánh giá trên Validation set (không cập nhật trọng số)
        ↓
8. Cosine Annealing giảm learning rate
        ↓
9. Lưu checkpoint nếu val_loss tốt hơn trước
```

**Siêu tham số:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| Batch size | 2.048 | Gradient ổn định, tận dụng bộ nhớ |
| Learning rate | 10⁻³ | Mặc định tốt cho AdamW |
| Weight decay | 10⁻⁴ | Tránh overfitting |
| Max epochs | 30 | Giới hạn trên |
| Early stopping | 5 epochs | Dừng khi val_loss không cải thiện |
| Optimizer | AdamW | Tốt hơn Adam về regularization |
| Scheduler | Cosine Annealing | Học nhanh đầu, tinh chỉnh cuối |

---

## SLIDE 6 — Kết Quả & Ứng Dụng

### Tiêu đề: Sau khi huấn luyện — Encoder được dùng như thế nào?

**Sau training, Decoder bị loại bỏ hoàn toàn.**

```
1,2 triệu bài hát
        ↓  Encoder f_θ (đã train)
1,2 triệu vector 32D
        ↓  L2-normalize
        ↓  FAISS IndexFlatIP
FAISS Index (tìm kiếm cosine similarity <1ms)
```

**Khi người dùng upload ảnh:**

```
Ảnh → CLIP → nhãn cảm xúc ("Happy")
           ↓
   AUDIO_PROFILES["Happy"] = [0.75, 0.72, 0.85, ...]
           ↓  Encoder
   Query vector 32D
           ↓  FAISS.search(top_k=10)
   10 bài hát gần nhất → Gợi ý
```

**Tại sao Latent Space 32D hoạt động?**

- Bài hát có âm thanh tương tự → encoder ra vector **gần nhau** trong không gian 32D
- FAISS đo **cosine similarity** → tìm bài hát "cùng âm sắc"
- Không cần nhãn "giống/khác" — cấu trúc tự hình thành trong quá trình học

---

## TÓM TẮT 1 TRANG

### So sánh Supervised vs Unsupervised (Model 1 vs Model 2)

| | **Model 1 (Hybrid)** | **Model 2 (Autoencoder)** |
|---|---|---|
| **Phương pháp** | Supervised Learning | Unsupervised Learning |
| **Biến phụ thuộc** | Nhãn emotion (vui/buồn...) | Chính đầu vào $\mathbf{x}$ |
| **Cần nhãn không?** | Có — gán thủ công | Không |
| **Hàm mất mát** | CrossEntropy / MSE với nhãn | $\text{MSE}(\hat{\mathbf{x}}, \mathbf{x})$ |
| **Dữ liệu** | 551K bài (có lyrics, emotion) | 1,2M bài (chỉ audio features) |
| **Đặc trưng đầu vào** | Audio + BERT lyrics (768D) | Audio only (9D) |
| **Latent space** | — | 32D (cosine similarity) |
| **Điểm mạnh** | Hiểu ngữ nghĩa lời bài hát | Mở rộng với dữ liệu lớn hơn |

---

*File này dùng để làm slide thuyết trình. Mỗi section "## SLIDE N" tương ứng với một trang slide.*
