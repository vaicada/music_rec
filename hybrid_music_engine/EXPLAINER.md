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

Module này có 2 class chính:

### FAISSIndex — Quản lý chỉ mục tìm kiếm

```python
class FAISSIndex:
    def create_index(self, embeddings, song_data):
        """Tạo FAISS index từ ma trận embedding."""
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # IndexFlatL2: tìm kiếm chính xác 100%, dùng khoảng cách L2
        cpu_index = faiss.IndexFlatL2(self.embedding_dim)
        cpu_index.add(embeddings)

        self.index = cpu_index
        self._build_mappings(song_data)  # Xây dựng bảng index → info bài hát

    def search(self, query_embedding, top_k=10):
        """Tìm top-k bài tương tự nhất."""
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        return [(idx, float(dist), self.idx_to_song[idx])
                for dist, idx in zip(distances[0], indices[0]) if idx >= 0]

    def save(self, path):
        faiss.write_index(self.index, str(path))
        # Lưu kèm mappings (index_pos → song info)
        pickle.dump({'song_ids': self.song_ids, 'idx_to_song': self.idx_to_song},
                    open(str(path) + '.mappings.pkl', 'wb'))
```

### MusicRecommendationEngine — API gợi ý cấp cao

```python
class MusicRecommendationEngine:
    def get_similar_songs(self, song_name, artist_name=None, top_k=5):
        # Bước 1: Tìm bài hát trong database
        song_row = self._find_song(song_name, artist_name)

        # Bước 2: Tối ưu — lấy vector từ FAISS (không cần encode lại)
        try:
            song_idx   = int(song_row.name)
            embedding  = self.faiss_index.reconstruct(song_idx)
        except:
            # Fallback: encode từ đầu nếu reconstruct thất bại
            embedding = self._encode_song(song_row).numpy().squeeze()

        # Bước 3: Tìm kiếm FAISS
        candidates = self.faiss_index.search(embedding, top_k * 3 + 1)

        # Bước 4: Lọc theo cảm xúc tương thích
        compatible_emotions = {
            'joy':     ['joy', 'love', 'surprise', 'anger'],
            'sadness': ['sadness', 'love', 'fear'],
            'anger':   ['anger', 'joy', 'fear'],
            'love':    ['love', 'joy', 'sadness'],
        }
        allowed = compatible_emotions.get(song_row.get('emotion', '').lower(), [])

        recommendations = []
        for idx, distance, song_info in candidates:
            if song_info['song'].lower() == song_name.lower(): continue  # Bỏ qua chính nó
            if allowed and song_info.get('emotion', '') not in allowed: continue

            similarity = 1.0 / (1.0 + distance)   # Chuyển L2 distance → similarity
            recommendations.append({
                'song': song_info['song'], 'artist': song_info['artist'],
                'similarity': round(similarity, 4),
                'emotion': song_info.get('emotion', '')
            })
            if len(recommendations) >= top_k: break

        return recommendations

    def _encode_song(self, song_row):
        """Encode một bài hát thành vector 64D."""
        # Lấy và làm sạch lời
        lyrics       = self.text_processor.clean_lyrics(song_row.get('text', ''))
        combined_txt = self.text_processor.combine_text_features(lyrics, ...)
        text_encoded = self.text_processor.tokenize_single(combined_txt)

        # Lấy đặc trưng âm thanh
        audio_features = self.audio_processor.transform(pd.DataFrame([song_row]))

        # Chạy qua model → lấy embedding 64D
        with torch.no_grad():
            embedding = self.model.get_embedding(
                input_ids=text_encoded['input_ids'].to(device),
                attention_mask=text_encoded['attention_mask'].to(device),
                audio_features=audio_features.to(device),
            )
        return embedding.cpu()
```

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
