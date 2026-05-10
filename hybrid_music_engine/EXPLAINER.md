 # 🧠 Giải thích Module `hybrid_music_engine/` — Model 1: BERT Hybrid

> Module này là "bộ não" của Model 1 — kết hợp BERT (hiểu lời bài hát) với đặc trưng âm thanh để gợi ý nhạc.

---

## Tổng quan các file

| File | Vai trò |
|---|---|
| `config.py` | Cấu hình tập trung (đường dẫn, siêu tham số) |
| `model.py` | Kiến trúc mạng Hybrid (BERT + Audio + Fusion) |
| `processors.py` | Xử lý dữ liệu: text, audio, metadata |
| `inference.py` | Engine gợi ý + FAISS search |
| `trainer.py` | Vòng lặp huấn luyện mô hình |
| `image_processor.py` | Phân loại ảnh → nhãn cảm xúc (CLIP) |
| `logger.py` | Logging có cấu trúc cho toàn hệ thống |

---

## 📄 File 1: `model.py` — Kiến trúc mạng Hybrid

### Ý tưởng thiết kế

Model 1 kết hợp **2 nguồn thông tin** về một bài hát:
- **Lời bài hát** → hiểu qua BERT (ngôn ngữ học)
- **Đặc trưng âm thanh** → 9 số đo (năng lượng, nhịp điệu, v.v.)

Hai luồng được **ghép lại (fusion)** thành một vector 64 chiều đại diện cho bài hát.

```
                    [BERT branch]                [Audio branch]
Lời bài hát ──► BertModel (768D) ──► Projection ──► 256D ──┐
                                                             ├──► Fusion → 64D embedding
Audio (9D) ──────────────────────► AudioEncoder ──► 64D ───┘
```

```python
class HybridMusicModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        # 1. BERT làm feature extractor (đông lạnh trọng số, chỉ lấy output)
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # 2. BERT Projection: thu nhỏ 768D → 256D qua 3 bước
        self.bert_proj = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(384, 256), nn.LayerNorm(256)
        )

        # 3. Audio Encoder: mở rộng 9D → 64D để học đặc trưng sâu hơn
        self.audio_enc = nn.Sequential(
            nn.Linear(9, 64),   nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.LayerNorm(64)
        )

        # 4. Fusion: ghép [256D + 64D] = 320D → nén dần → 64D
        self.fusion = nn.Sequential(
            nn.Linear(320, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64)
        )

        # 5. Các đầu phân loại (dùng khi training)
        self.emotion_head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 6))
        self.genre_head   = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 3010))

    def forward(self, input_ids, attention_mask, audio_features, token_type_ids=None):
        # Bước 1: BERT lấy [CLS] token embedding (đông lạnh gradient)
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            bert_emb = bert_out.last_hidden_state[:, 0, :]  # shape: [B, 768]

        # Bước 2: Chiếu BERT xuống 256D
        b = self.bert_proj(bert_emb)          # shape: [B, 256]

        # Bước 3: Encode audio features lên 64D
        a = self.audio_enc(audio_features)    # shape: [B, 64]

        # Bước 4: Ghép và fusion → 64D embedding
        combined  = torch.cat([b, a], dim=-1) # shape: [B, 320]
        embedding = self.fusion(combined)      # shape: [B, 64]

        # L2-normalize để dùng cosine similarity với FAISS
        embedding = F.normalize(embedding, p=2, dim=-1)

        return {'embedding': embedding}
```

**🔍 Giải thích chi tiết Kiến trúc Model:**
- **`nn.Linear` + `nn.GELU`**: Các lớp mạng nơ-ron truyền thẳng (Feed-Forward) sử dụng hàm kích hoạt GELU. GELU mượt mà hơn ReLU, thường cho hiệu suất tốt hơn trong các bài toán NLP và âm thanh hiện đại.
- **`nn.LayerNorm` & `nn.BatchNorm1d`**: Chuẩn hóa dữ liệu ở mỗi tầng. Điều này giúp ngăn chặn triệt tiêu/bùng nổ gradient và giúp mô hình hội tụ nhanh hơn.
- **`nn.Dropout(0.15 - 0.3)`**: Ngẫu nhiên vô hiệu hóa một số liên kết trong quá trình training. Điều này ép mô hình không được phụ thuộc vào 1 nơ-ron cụ thể nào, từ đó chống *overfitting* (học vẹt).
- **`F.normalize(embedding, p=2, dim=-1)`**: Đây là bước cực kỳ quan trọng! Việc chuẩn hóa L2 đưa vector về độ dài bằng 1. Khi vector đã chuẩn hóa L2, việc tính khoảng cách Euclid (bằng FAISS) sẽ tương đương về mặt toán học với việc tính **Cosine Similarity** — thước đo chuẩn nhất để so sánh sự tương đồng của văn bản/âm thanh.

**Luồng dữ liệu qua hàm `forward`:**
- **Bước 1**: Đóng băng BERT (`torch.no_grad()`) và lấy vector `[CLS]` 768 chiều (đại diện cho ngữ nghĩa toàn bộ lời bài hát).
- **Bước 2**: Đưa vector 768D qua mạng `bert_proj` để nén thông tin ngôn ngữ xuống còn 256 chiều.
- **Bước 3**: Đưa 9 đặc trưng âm thanh gốc qua `audio_enc` để mở rộng lên không gian 64 chiều, giúp mô hình bắt được các điểm tương đồng phức tạp.
- **Bước 4**: Ghép nối (concatenate) 2 luồng: `256D (text) + 64D (audio) = 320D`. Sau đó đẩy qua mạng `fusion` để nén lần cuối ra **vector 64D duy nhất** đại diện cho toàn bộ bài hát. Cuối cùng, chuẩn hóa L2 vector này trước khi trả về.

---

## 📄 File 2: `processors.py` — Xử lý dữ liệu (3 Processor)

### TextProcessor — Xử lý lời bài hát

```python
class TextProcessor:
    def clean_lyrics(self, text):
        """Làm sạch lời bài hát."""
        text = re.sub(r'https?://\S+', '', text)    # Xóa URL
        text = re.sub(r'<[^>]+>', '', text)          # Xóa thẻ HTML
        text = re.sub(r'\[.*?\]', '', text)          # Xóa [Verse], [Chorus]
        text = text.lower()
        text = re.sub(r'[^\w\s\'\"\.\\,\!\?]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def combine_text_features(self, lyrics, emotion="", genre=""):
        """Ghép lời với context → BERT hiểu rõ hơn ngữ cảnh."""
        parts = []
        if emotion: parts.append(f"[EMOTION: {emotion}]")  # vd: "[EMOTION: joy]"
        if genre:   parts.append(f"[GENRE: {genre}]")      # vd: "[GENRE: pop]"
        if lyrics:  parts.append(lyrics)
        return " ".join(parts)
        # Kết quả: "[EMOTION: joy] [GENRE: pop] i love you baby..."

    def tokenize_batch(self, texts):
        """Tokenize danh sách văn bản cho BERT."""
        encoded = self.tokenizer(
            texts,
            max_length=256,
            padding='max_length',  # Pad về đúng 256 token
            truncation=True,       # Cắt nếu quá dài
            return_tensors="pt"
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
```

### AudioProcessor — Chuẩn hóa đặc trưng âm thanh

```python
class AudioProcessor:
    def fit(self, data):
        """Tính mean và std từ tập TRAIN (tránh data leakage)."""
        features   = data[self.feature_names].values.astype(np.float32)
        self._mean = np.mean(features, axis=0)
        self._std  = np.std(features, axis=0) + 1e-8  # +1e-8 tránh chia 0
        return self

    def transform(self, data):
        """Z-score normalize: (x - mean) / std"""
        features   = data[self.feature_names].values.astype(np.float32)
        normalized = (features - self._mean) / self._std
        return torch.tensor(normalized, dtype=torch.float32)

    def save_stats(self, path):
        """Lưu stats để dùng khi inference (web app không train lại)."""
        stats = {"mean": self._mean.tolist(), "std": self._std.tolist(),
                 "features": self.feature_names}
        json.dump(stats, open(path, 'w'))
```

**🔍 Giải thích chi tiết AudioProcessor:**
- Tại sao dùng **Z-score Normalization `(x - mean) / std`** thay vì Min-Max (0-1)? Các đặc trưng âm thanh như `tempo` (nhịp độ) hay `loudness` (âm lượng) thường chứa các giá trị ngoại lệ (outlier). Chuẩn hóa Z-score xử lý outlier tốt hơn và giữ nguyên phân phối gốc của dữ liệu.
- Lưu ý kỹ thuật `+ 1e-8` khi tính `std` là một "trick" tiêu chuẩn để tránh lỗi Crash do chia cho 0 (khi một cột đặc trưng có tất cả các giá trị hoàn toàn bằng nhau).

### MetadataProcessor — Encode dữ liệu categorical

```python
class MetadataProcessor:
    def fit(self, data):
        """Xây dựng từ điển genre → số và emotion → số."""
        genres = data['genre'].dropna().unique()
        self.genre_to_idx = {g: i for i, g in enumerate(genres)}
        self.genre_to_idx['<UNK>'] = len(self.genre_to_idx)  # token không xác định

        emotions = data['emotion'].dropna().unique()
        self.emotion_to_idx = {e: i for i, e in enumerate(emotions)}

    def transform_genre(self, genres):
        """'pop' → 0, 'rock' → 1, ..."""
        unk = self.genre_to_idx.get('<UNK>', 0)
        return torch.tensor([self.genre_to_idx.get(g, unk) for g in genres])

    def transform_emotion(self, emotions):
        unk = self.emotion_to_idx.get('<UNK>', 0)
        return torch.tensor([self.emotion_to_idx.get(e, unk) for e in emotions])
```

### MusicDataset — Kết hợp 3 processor

```python
class MusicDataset(Dataset):
    """Đưa một dòng dữ liệu qua cả 3 processor để tạo sample cho model."""

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Text: lời + emotion + genre → tokenized
        lyrics       = self.text_processor.clean_lyrics(row["text"])
        combined_txt = self.text_processor.combine_text_features(lyrics, row["emotion"], row["genre"])
        text_encoded = self.text_processor.tokenize_single(combined_txt)

        # Audio: đã normalize sẵn
        audio = self.audio_features[idx]

        # Metadata: convert categorical → số
        genre_idx   = self.metadata_processor.transform_genre([row["genre"]])[0]
        emotion_idx = self.metadata_processor.transform_emotion([row["emotion"]])[0]

        return {
            "input_ids":      text_encoded["input_ids"].squeeze(0),
            "attention_mask": text_encoded["attention_mask"].squeeze(0),
            "audio_features": audio,
            "emotion_label":  emotion_idx,  # Dùng để supervise emotion head
            "genre_idx":      genre_idx,
        }
```

---

## 📄 File 3: `inference.py` — Engine gợi ý + FAISS

Module này chịu trách nhiệm biến các vector đặc trưng (embeddings) được sinh ra từ mô hình thành một hệ thống tìm kiếm và gợi ý âm nhạc thực tế. Khác với giai đoạn huấn luyện (chạy chậm để học), hệ thống này phải đảm bảo tốc độ phản hồi tính bằng mili-giây khi người dùng yêu cầu trên Web App. Module có 2 class chính:

### FAISSIndex — Quản lý chỉ mục tìm kiếm tốc độ cao

FAISS (Facebook AI Similarity Search) là một thư viện của Meta dùng để tìm kiếm các vector giống nhau cực kỳ nhanh. Nếu không có FAISS, để tìm 1 bài giống với "Shape of You", bạn phải so sánh vector của nó với toàn bộ 500.000 bài hát khác (rất chậm). FAISS tổ chức lại các vector này vào một cấu trúc chỉ mục (index) giúp tìm kiếm tức thời.

```python
class FAISSIndex:
    def create_index(self, embeddings, song_data):
        """Tạo FAISS index từ ma trận embedding."""
        # Chuyển đổi dữ liệu sang định dạng C-contiguous (liên tục trong bộ nhớ RAM)
        # Đây là yêu cầu bắt buộc của FAISS vì thư viện lõi viết bằng C++
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # IndexFlatL2: Cấu trúc tìm kiếm vét cạn (brute-force) nhưng được tối ưu hóa SIMD
        # Đảm bảo tìm kiếm chính xác 100%, dùng khoảng cách L2 (Euclid)
        # Do trước đó embeddings đã được L2-normalize, khoảng cách này tỷ lệ nghịch với Cosine Similarity
        cpu_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Thêm toàn bộ các vector bài hát vào chỉ mục
        cpu_index.add(embeddings)

        self.index = cpu_index
        # Xây dựng bảng ánh xạ (mappings) giữa số thứ tự index và thông tin thực của bài hát
        self._build_mappings(song_data)

    def search(self, query_embedding, top_k=10):
        """Tìm top-k bài tương tự nhất."""
        # FAISS search trả về 2 mảng:
        # 1. distances: khoảng cách L2 từ câu truy vấn đến kết quả
        # 2. indices: vị trí (ID) của bài hát trong index
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        # Biến đổi kết quả ID thành thông tin bài hát thực tế
        return [(idx, float(dist), self.idx_to_song[idx])
                for dist, idx in zip(distances[0], indices[0]) if idx >= 0]

    def reconstruct(self, idx):
        """
        Lấy lại vector 64D của bài hát trực tiếp từ RAM của FAISS.
        Tránh việc phải gọi qua Model sinh lại.
        """
        return self.index.reconstruct(int(idx))
        
    def save(self, path):
        # Lưu cấu trúc FAISS thành file nhị phân (.bin) siêu nhẹ
        faiss.write_index(self.index, str(path))
        # Lưu kèm file Mappings (.pkl) để tra cứu tên bài hát
        pickle.dump({'song_ids': self.song_ids, 'idx_to_song': self.idx_to_song},
                    open(str(path) + '.mappings.pkl', 'wb'))
```

**🔍 Giải thích Kỹ thuật FAISS:**
- **Tại sao lại ép kiểu `np.ascontiguousarray(embeddings.astype(np.float32))`?** Các mảng Numpy trong Python có thể lưu trữ rải rác trong bộ nhớ (fragmented). FAISS là một thư viện C++ hiệu năng cao, nó đòi hỏi dữ liệu phải nằm thành một khối bộ nhớ liên tục (contiguous) kiểu số thực 32-bit (float32). Lệnh này dọn dẹp và chuẩn bị bộ nhớ để nạp vào lõi C++ của FAISS.
- **`IndexFlatL2`**: Là kiểu Index tính khoảng cách Euclid (L2) chuẩn xác nhất. Dù FAISS có nhiều loại Index khác như HNSW hay IVFPQ (nhanh hơn nhưng có sai số), với 500k bài hát, `IndexFlatL2` vẫn đáp ứng tốc độ vài mili-giây mà lại không làm giảm chất lượng gợi ý.
- **Mapping (Ánh xạ)**: FAISS chỉ biết các vector dưới dạng ID từ `0` đến `499,999`. Nó hoàn toàn không biết ID `1` là bài "Shape of You". Do đó, ta cần một bộ từ điển `idx_to_song` đi kèm để sau khi FAISS trả về ID, ta "dịch" nó lại thành tên bài hát.

### MusicRecommendationEngine — API gợi ý cấp cao

Đây là Class gom tất cả lại: Model + FAISS + Xử lý Logic Nghiệp Vụ, cung cấp hàm giao tiếp cuối cùng cho Web Server.

```python
class MusicRecommendationEngine:
    def get_similar_songs(self, song_name, artist_name=None, top_k=5):
        # Bước 1: Tìm bài hát trong database (Bằng hàm fuzzy/exact match tự viết)
        song_row = self._find_song(song_name, artist_name)

        # Bước 2: Tối ưu Tốc Độ Nhất — lấy vector gốc từ RAM
        try:
            # Nếu bài hát đã có sẵn trong tập data 500k bài,++
            # Ta KHÔNG cần chạy qua mô hình AI chậm chạp,
            # Chỉ cần rút vector 64D thẳng từ bộ nhớ FAISS
            song_idx   = int(song_row.name)
            embedding_np  = self.faiss_index.reconstruct(song_idx)
        except:
            # Fallback: Chỉ khi bài hát mới 100% người dùng tự nhập (không có trong data)
            # Thì mới gọi BERT và Neural Net để dịch từ đầu mất ~100ms
            embedding = self._encode_song(song_row)
            embedding_np = embedding.numpy().squeeze()

        # Bước 3: Tìm kiếm trên FAISS (Tìm nhiều hơn cần thiết 3 lần để phòng hờ lọc)
        candidates = self.faiss_index.search(embedding_np, top_k * 3 + 1)

        # Bước 4: Lọc theo cảm xúc tương thích (Luật Hard-code)
        # Tránh trường hợp đang nghe nhạc Vui (Joy) lại gợi ý nhạc Tức Giận (Anger)
        query_emotion = song_row.get('emotion', '').lower()
        compatible_emotions = {
            'joy':     ['joy', 'love', 'surprise', 'anger'],
            'sadness': ['sadness', 'love', 'fear'],
            'anger':   ['anger', 'joy', 'fear'],
            'love':    ['love', 'joy', 'sadness'],
            'fear':    ['fear', 'sadness', 'anger'],
        }
        allowed = compatible_emotions.get(query_emotion, [])

        recommendations = []
        for idx, distance, song_info in candidates:
            # Bỏ qua chính bài hát gốc đang truy vấn (Vd: Search "Shape of you" thì ko gợi ý lại "Shape of you")
            if str(song_info['song']).lower() == song_name.lower(): continue  
            
            # Nếu vi phạm luật cảm xúc -> Loại bài này
            if allowed and song_info.get('emotion', '').lower() not in allowed: continue

            # Chuyển Khoảng cách L2 (Càng nhỏ càng giống) thành Điểm Tương Đồng (Càng lớn càng giống, Max = 1.0)
            similarity = 1.0 / (1.0 + distance)   
            
            recommendations.append({
                'song': song_info['song'], 
                'artist': song_info['artist'],
                'similarity': round(similarity, 4),
                'emotion': song_info.get('emotion', '')
            })
            # Lấy đủ top_k bài thì dừng
            if len(recommendations) >= top_k: break

        return recommendations

    def _encode_song(self, song_row):
        """Dịch 1 bài hát chưa từng xuất hiện thành vector 64D (Rất chậm)."""
        # Bước A: Làm sạch và chuẩn bị Lời bài hát
        lyrics       = self.text_processor.clean_lyrics(song_row.get('text', ''))
        emotion      = str(song_row.get('emotion', ''))
        genre        = str(song_row.get('genre', ''))
        combined_txt = self.text_processor.combine_text_features(lyrics, emotion, genre)
        text_encoded = self.text_processor.tokenize_single(combined_txt)

        # Bước B: Xử lý 9 chỉ số âm thanh
        audio_features = self.audio_processor.transform(pd.DataFrame([song_row]))

        # Bước C: Đẩy qua Mô hình Hybrid
        with torch.no_grad(): # Tắt tính toán đạo hàm (Gradient) để tăng tốc bộ nhớ
            embedding = self.model.get_embedding(
                input_ids=text_encoded['input_ids'].to(self.device),
                attention_mask=text_encoded['attention_mask'].to(self.device),
                audio_features=audio_features.to(self.device),
            )
        return embedding.cpu() # Đưa vector từ GPU RAM về System RAM
```

**🔍 Giải thích chi tiết Engine Gợi ý (Logic Nghiệp vụ):**
- **Sự lợi hại của `reconstruct()`**: Thay vì phải đẩy bài hát gốc qua toàn bộ mạng Neural Network 110 triệu tham số của BERT, hàm `reconstruct(song_idx)` móc vector 64D trực tiếp trong RAM (đã được lưu sẵn trong lúc Build Index). Tốc độ lấy là $O(1)$ — chưa tới 1 mili-giây. Đây là kỹ thuật sống còn để triển khai ứng dụng thực tế.
- **Cơ chế Fallback (`except`)**: Nếu người dùng nhập vào một lời bài hát tự chế, nó không nằm trong 500,000 bài có sẵn, lúc đó hàm `_encode_song` mới được gọi. Hàm này thực hiện đúng chuỗi `Làm Sạch -> Tokenize -> Audio Transform -> Model Forward`. Do dùng `torch.no_grad()`, nó tốn ít RAM nhưng thời gian sẽ mất khoảng ~100-200ms.
- **Lọc cảm xúc cứng (Rule-based Filtering)**: Các thuật toán AI đôi lúc rất "ngu ngốc". Một bài hát có giai điệu vui nhưng lời nhắc tới cái chết có thể bị xếp nhầm vector gần một bài nhạc buồn. Để bảo vệ trải nghiệm cảm xúc của người dùng, ta áp dụng một lớp luật *hard-code* (`compatible_emotions`). Lớp luật này cấm ngặt việc chuyển đột ngột từ nhạc Sadness sang nhạc Anger, đóng vai trò như một màng lọc an toàn.
- **Công thức `similarity = 1.0 / (1.0 + distance)`**: Trong FAISS `IndexFlatL2`, nếu hai vector y hệt nhau, khoảng cách = `0`. Nhưng con người thì thích xem "Độ tương đồng 100%". Công thức trên biến số `0` thành `1.0` (Tương đồng tối đa), và biến khoảng cách xa (ví dụ `3.0`) thành một số điểm nhỏ dần (ví dụ `0.25`).

---

## 🗺️ Sơ đồ luồng dữ liệu

```
[TRAINING - offline]
train.csv ──► DataManager ──► MusicDataset ──► DataLoader
                                  │
              TextProcessor ───────┤
              AudioProcessor ──────┤──► HybridMusicModel.forward()
              MetadataProcessor ───┘          │
                                         MSE + CrossEntropy Loss
                                              │
                                    best_model.pth

[BUILD INDEX - offline]
best_model.pth + song_metadata.csv
    ↓ MusicRecommendationEngine.build_index()
    ↓ encode 551K bài → 64D embeddings
    ↓ FAISSIndex.create_index()
    → faiss_index.bin + faiss_index.bin.mappings.pkl

[INFERENCE - online, web app gọi]
"Shape of You" ──► _find_song() ──► FAISSIndex.reconstruct()
                                          ↓
                                    FAISS.search(top_k=10)
                                          ↓
                               lọc emotion compatible
                                          ↓
                               [10 gợi ý + similarity score]
```
