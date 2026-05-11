# Huấn luyện Model 1 — Hybrid Music Recommender

> **Dành cho thuyết trình / slide**
> Mỗi section `## SLIDE N` tương ứng một trang slide.

---

## SLIDE 1 — Tổng quan

### Tiêu đề: Model 1: Học Có Giám Sát với Kiến trúc Hybrid Đa Nhánh

**Vấn đề đặt ra:**

- Âm nhạc mang ý nghĩa từ **nhiều chiều**: lời bài hát (ngữ nghĩa), âm thanh (đặc trưng số), cảm xúc (nhãn)
- Một nhánh mạng đơn lẻ không thể nắm bắt đủ thông tin
- Cần một mô hình **học cùng lúc nhiều tác vụ** (Multi-task Learning)

**Giải pháp:**

- Kiến trúc **Hybrid 2 nhánh**: nhánh BERT (lời bài hát) + nhánh Audio (đặc trưng âm thanh)
- Phương pháp: **Supervised Learning** kết hợp **Metric Learning (Triplet Loss)**
- Mục tiêu: học một **không gian nhúng 64 chiều** nơi bài hát tương tự nằm gần nhau

---

## SLIDE 2 — Dữ Liệu Đầu Vào

### Tiêu đề: Dữ liệu đầu vào — Đa phương thức (Multimodal)

**Mỗi bài hát trong tập huấn luyện có 3 loại thông tin:**

| Loại | Dữ liệu | Chiều |
|---|---|---|
| **Lời bài hát** | Text (lyrics) — tokenized bởi BERT | 256 tokens |
| **Âm thanh** | 9 đặc trưng Spotify API | 9 số thực |
| **Nhãn cảm xúc** | 6 nhãn: joy, sadness, anger, fear, love, surprise | 1 số nguyên |

**Nguồn dữ liệu:**

- **551K bài hát** từ `final_milliondataset_BERT_500K_revised.json`
- Đã có sẵn nhãn emotion (được gán bởi mô hình NLP hoặc chuyên gia)
- Chia tập: Train 80% / Val 10% / Test 10%

---

## SLIDE 3 — Kiến Trúc Mô Hình

### Tiêu đề: Kiến trúc Hybrid — 2 Nhánh + Fusion Layer

```
                    MỖI BÀI HÁT
                         │
          ┌──────────────┴──────────────┐
          │                             │
   [Lyrics Text]                 [9 Audio Features]
          │                             │
    ┌─────▼──────┐               ┌──────▼──────┐
    │    BERT    │               │ Audio Branch │
    │ (frozen)   │               │              │
    │ 768D → CLS │               │  9 → 64      │
    └─────┬──────┘               │  64 → 128    │
          │                      │  128 → 64    │
    BERT Projection              └──────┬───────┘
    768 → 512 → 384 → 256D             │ 64D
          │                             │
          └──────────┬──────────────────┘
                     │ Concat (256 + 64 = 320D)
               ┌─────▼──────┐
               │  Fusion    │
               │ 320 → 256  │
               │ 256 → 192  │
               │ 192 → 128  │
               │ 128 → 64   │
               └─────┬──────┘
                     │ L2-normalize
                ┌────▼────┐
                │ 64D     │  ← Embedding cuối cùng
                │ Embedding│    (dùng cho FAISS)
                └────┬────┘
          ┌──────────┤
    ┌─────▼─────┐  ┌─▼──────────┐
    │ Emotion   │  │ Genre Head │
    │ Head      │  │ 64→128→N   │
    │ 64→32→6   │  └────────────┘
    └───────────┘
```

**Nhánh BERT:**

- `bert-base-uncased` — 12 layers, 768D hidden size
- **10/12 layers bị freeze** — chỉ fine-tune 2 layers cuối
- Lấy embedding của token `[CLS]` làm đại diện cho lời bài hát

**Nhánh Audio:**

- Input: 9 đặc trưng âm thanh (energy, danceability, valence, tempo...)
- Kiến trúc: `Linear(9→64) → BatchNorm → GELU → Linear(64→128) → BatchNorm → GELU → Linear(128→64)`

---

## SLIDE 4 — Hàm Mất Mát (Multi-task Loss)

### Tiêu đề: Học Đồng Thời 2 Mục Tiêu — Triplet + Emotion

**Tổng loss là tổ hợp có trọng số của 2 thành phần:**

$$\mathcal{L}_{total} = \underbrace{0.5 \cdot \mathcal{L}_{triplet}}_{\text{Metric Learning}} + \underbrace{0.3 \cdot \mathcal{L}_{emotion}}_{\text{Classification}}$$

---

**① Triplet Loss — học không gian tương đồng**

$$\mathcal{L}_{triplet} = \max\!\left(0,\ \|f(a) - f(p)\|_2^2 - \|f(a) - f(n)\|_2^2 + \text{margin}\right)$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $f(a)$ | Embedding của bài hát anchor (ví dụ: "Bohemian Rhapsody") |
| $f(p)$ | Embedding của bài hát positive — **cùng thể loại/cảm xúc** |
| $f(n)$ | Embedding của bài hát negative — **khác thể loại/cảm xúc** |
| margin | 0.3 — khoảng cách tối thiểu buộc phải duy trì |

> **Ý nghĩa:** Buộc mô hình đặt bài hát tương tự **gần nhau** và bài hát khác nhau **xa nhau** trong không gian 64D.

---

**② Emotion Loss — phân loại cảm xúc**

$$\mathcal{L}_{emotion} = \text{CrossEntropy}(\hat{y}_{emotion},\ y_{emotion})$$

> **Ý nghĩa:** Mô hình phải dự đoán đúng cảm xúc của bài hát (joy/sadness/anger/fear/love/surprise).
> Điều này **ép Emotion Head** học embedding có ý nghĩa cảm xúc.

---

**Triplet Mining — cách tạo bộ 3 (anchor, positive, negative):**

- **Positive**: bài hát tương tự trong dataset (cột `Similar Song 1/2/3`) hoặc cùng genre
- **Negative**: bài hát từ genre **khác** với anchor
- Nếu không tìm được positive → bỏ qua bộ đó (không train)

---

## SLIDE 5 — Quá Trình Huấn Luyện

Training Loop — Mỗi Epoch

```
Mỗi epoch (tối đa 20 epochs):

1. Lấy batch 32 bài hát (lyrics + audio + emotion label)
        ↓
2. BERT encode lyrics → CLS embedding 768D
        ↓
3. BERT Projection: 768D → 256D
   Audio Branch: 9D → 64D
        ↓
4. Fusion: Concat(256+64=320D) → 64D embedding
        ↓
5. Tính Triplet Loss (metric learning)
   Tính Emotion Loss (CrossEntropy)
   → Total Loss = 0.5×Triplet + 0.3×Emotion
        ↓
6. Backward pass: tính gradient
   Gradient Clipping (max norm = 1.0)
   AdamW cập nhật trọng số
   Linear Scheduler bước tiếp theo
        ↓
7. Validation: tính val_loss + emotion accuracy
        ↓
8. Nếu val_loss tốt hơn → lưu best_model.pth
   Nếu 5 epoch liên tiếp không cải thiện → Early Stopping
```

---

**Siêu tham số:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| Batch size | 32 | BERT nặng — bộ nhớ giới hạn |
| Learning rate (BERT) | 2×10⁻⁶ | BERT đã pretrained — fine-tune nhẹ (lr/10) |
| Learning rate (khác) | 2×10⁻⁵ | Các lớp mới khởi tạo — học nhanh hơn |
| Weight decay | 0.01 | L2 regularization tránh overfitting |
| Warmup steps | 1.000 | Tăng lr dần thay vì nhảy đột ngột |
| Scheduler | Linear Warmup | Ổn định hội tụ với BERT |
| Gradient clipping | 1.0 | Ngăn gradient explosion với BERT |
| Max epochs | 20 | Giới hạn trên |
| Early stopping | 5 epochs | Dừng khi val_loss không cải thiện |
| Triplet margin | 0.3 | Khoảng cách tối thiểu positive vs negative |

---

## SLIDE 6 — Điểm Đặc Biệt Kỹ Thuật

### Tiêu đề: 3 Quyết định Thiết Kế Quan Trọng

**① Differential Learning Rate — BERT vs các lớp khác**
> BERT đã được pre-trained trên hàng tỷ từ — không cần thay đổi nhiều.
> Lớp Fusion và Audio Branch học từ đầu — cần lr lớn hơn 10×.

```
BERT layers:      lr = 2e-6  (nhỏ — fine-tune nhẹ)
Projection/Fusion: lr = 2e-5  (lớn — học nhanh hơn)
```

**② Gradient Clipping (max norm = 1.0)**
> BERT có nhiều lớp → gradient có thể bùng nổ (exploding gradient).
> Clipping giới hạn tổng độ lớn gradient, giúp training ổn định.

**③ Multi-task Learning — tại sao train cả 2 loss cùng lúc?**
>
> - Chỉ Triplet Loss: embedding học cụm tốt nhưng không có ý nghĩa cảm xúc rõ ràng
> - Chỉ Emotion Loss: phân loại tốt nhưng embedding không tối ưu cho similarity search
> - **Kết hợp cả hai**: embedding vừa **có cấu trúc không gian** (Triplet) vừa **có ý nghĩa ngữ nghĩa** (Emotion)

---

## SLIDE 7 — Kết Quả & Ứng Dụng

### Tiêu đề: Sau Huấn Luyện — Embedding được dùng như thế nào?

**Sau training, toàn bộ 551K bài hát được encode:**

```
551K bài hát (lyrics + audio)
        ↓  HybridMusicModel.forward()
551K vectors 64D (L2-normalized)
        ↓  FAISS IndexFlatL2
FAISS Index → tìm kiếm exact search
```

**Khi người dùng gửi query:**

```
Query (mood/context/image) → nhãn cảm xúc
           ↓
   Tìm bài hát có emotion khớp trong dataset
           ↓
   Encode qua HybridMusicModel → 64D query vector
           ↓
   FAISS.search(top_k=10)
           ↓
   10 bài hát gần nhất theo cosine similarity
```

---

## TÓM TẮT SO SÁNH — Model 1 vs Model 2

| | **Model 1 (Hybrid)** | **Model 2 (Autoencoder)** |
|---|---|---|
| **Phương pháp** | Supervised + Metric Learning | Unsupervised |
| **Dữ liệu đầu vào** | Lyrics (text) + 9 audio features | 9 audio features |
| **Cần nhãn không?** | **Có** — nhãn emotion | **Không** |
| **Hàm mất mát** | Triplet Loss + CrossEntropy | MSE(reconstruction) |
| **Kích thước embedding** | **64D** | **32D** |
| **Dữ liệu train** | 551K bài (có lyrics) | 1,2M bài |
| **Điểm mạnh** | Hiểu ngữ nghĩa lời bài hát, phân loại cảm xúc | Mở rộng quy mô dễ, không cần nhãn |
| **FAISS index** | IndexFlatL2 (L2 distance) | IndexFlatIP (cosine similarity) |

---

*File này dùng để làm slide thuyết trình. Mỗi section `## SLIDE N` tương ứng một trang slide.*
