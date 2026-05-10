# 🖼️ Image Processor Explainer

Tài liệu này giải thích chi tiết cách hệ thống gợi ý bài hát từ hình ảnh hoạt động, đặc biệt trả lời câu hỏi: **"Ảnh khác nhau nhưng cùng một cảm xúc (mood) thì bài hát gợi ý ra có giống nhau không? Tại sao?"**

---

## 1. Pipeline Xử lý Hình ảnh (Image to Music)

Tính năng gợi ý nhạc từ hình ảnh được xử lý qua 2 bước chính:

1. **Phân tích hình ảnh (Image Analysis):** Sử dụng model **CLIP** của OpenAI (qua `ImageMoodClassifier`).
2. **Gợi ý bài hát (Music Recommendation):** Sử dụng nhãn (label) kết quả để tìm bài hát qua **Model 1** hoặc **Model 2**.

### Bước 1: Ảnh → Label (CLIP Model)
File: `hybrid_music_engine/image_processor.py`

CLIP không tạo ra embedding trực tiếp để tìm kiếm bài hát. Thay vào đó, CLIP làm bài toán **Zero-shot Classification**:
- Định nghĩa sẵn 10 "Visual Prompts" (ví dụ: *"a photo of a happy smiling person"*, *"a peaceful sunset"*).
- Các prompts này được map sẵn vào **10 nhãn cố định** (5 Moods: Happy, Sad, Energetic, Calm, Angry | 5 Contexts: Party, Workout, Study, Relax, Driving).
- Khi người dùng tải ảnh lên, CLIP sẽ chọn prompt khớp nhất với ảnh, và trả về **nhãn chữ (text label)** tương ứng.

Ví dụ:
- Ảnh 1 (Một người cười tươi) → CLIP dự đoán "Happy".
- Ảnh 2 (Một chú chó nhảy lên vui vẻ) → CLIP dự đoán "Happy".

### Bước 2: Label → Bài hát

Label chữ này (ví dụ: "Happy") sau đó sẽ được truyền xuống Engine gợi ý. Tùy vào người dùng chọn Model 1 hay Model 2 mà cách xử lý sẽ khác nhau:

#### 🟢 Đối với Model 1 (Hybrid Music Engine)
File: `hybrid_music_engine/inference.py` (Hàm `get_recommendations_by_mood`)

- Label "Happy" được map vào các filter cố định: `{'emotion': ['joy'], 'valence': (0.6, 1.0), 'energy': (0.5, 1.0)}`.
- Engine sẽ lọc toàn bộ database bài hát theo các điều kiện này.
- Cuối cùng, Engine sắp xếp theo độ phổ biến (`Popularity`) và lấy Top 10.

#### 🔵 Đối với Model 2 (Audio Autoencoder)
File: `audio_model/clip_audio_bridge.py` (Hàm `recommend_from_label`)

- Label "Happy" được map thẳng vào một **Audio Profile cố định**: `[0.75, 0.72, 0.85, 120.0, 0.15, 0.05, 0.10, 0.15, 0.45]`.
- Mảng 9 đặc trưng âm thanh này luôn luôn giống nhau cho chữ "Happy".
- Sau đó, mảng này đi qua Autoencoder để tạo embedding 32 chiều và search FAISS lấy Top 10.

---

## 2. Trả lời câu hỏi trọng tâm

> **Câu hỏi:** "Bây giờ ảnh khác nhau gợi ý ra cùng mood thì bài hát có giống nhau không, nếu khác thì tại sao lại khác?"

**Trả lời:** **Có, bài hát gợi ý ra sẽ GIỐNG HỆT NHAU.**

### Tại sao lại giống hệt nhau?

Hệ thống hiện tại hoạt động theo cơ chế **Nút thắt cổ chai Text (Text Bottleneck)**:
1. Mọi bức ảnh, dù màu sắc hay chi tiết khác nhau thế nào, đều bị CLIP ép về **một trong 10 chữ cố định** (VD: "Happy", "Sad", "Relax").
2. Thông tin chi tiết của bức ảnh (biển xanh, núi đồi, mặt người) hoàn toàn **bị vứt bỏ** sau bước 1. Hệ thống chỉ giữ lại đúng 1 chữ duy nhất.
3. Từ 1 chữ "Happy", cả Model 1 và Model 2 đều sử dụng **logic tĩnh (static logic)**:
   - Model 1: Lọc bằng rule cố định + Sort theo Popularity. Lần nào sort cũng ra thứ tự đó.
   - Model 2: Chữ "Happy" tạo ra 1 vector audio cố định. Vector không đổi thì FAISS luôn search ra cùng 1 bộ nearest neighbors.

Vì vậy, nếu Ảnh A và Ảnh B đều được CLIP nhận diện là "Happy", **10 bài hát gợi ý cho Ảnh A sẽ hoàn toàn trùng khớp với 10 bài hát gợi ý cho Ảnh B.**

---

## 3. Cách khắc phục (Đã được áp dụng)

Để giải quyết vấn đề ảnh khác nhau (nhưng cùng mood) ra nhạc giống hệt nhau, hệ thống đã được **cập nhật để thêm tính ngẫu nhiên (Random Sampling)**:

1. **Đối với Model 1 (Hybrid):**
   - Thay vì chỉ lấy Top 10 bài phổ biến nhất, hệ thống giờ đây sẽ lọc ra **Top 100 bài phổ biến nhất** thỏa mãn mood.
   - Sau đó, lấy ngẫu nhiên (random sample) 10 bài từ danh sách 100 bài này.
   - Kết quả: Mỗi lần gọi API (dù cùng ảnh hay khác ảnh nhưng chung mood), bạn sẽ nhận được một danh sách 10 bài hát khác nhau, tạo sự đa dạng và bất ngờ cho người dùng.

2. **Đối với Model 2 (Audio Autoencoder):**
   - Tương tự, thay vì lấy chính xác Top 10 Nearest Neighbors từ FAISS.
   - Hệ thống sẽ query **Top 50 Nearest Neighbors** gần nhất với vector của mood đó.
   - Sau đó lấy ngẫu nhiên 10 bài từ top 50 này và sắp xếp lại theo độ phù hợp.

**Kết luận:** Nhờ có bước Random Sampling này, tính năng gợi ý theo hình ảnh (và theo mood/context) nay đã trở nên đa dạng và không còn bị lặp lại kết quả như trước!
