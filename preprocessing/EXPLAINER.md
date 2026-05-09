# 📁 Giải thích Thư mục `preprocessing/` — Toàn bộ Preprocessing Pipeline

> Đây là bước đầu tiên và quan trọng nhất: làm sạch và chuẩn bị dữ liệu thô trước khi AI có thể học.

---

## Tổng quan 3 file

| File | Dùng cho | Dataset đầu vào |
|---|---|---|
| `prepare_data.py` | Model 1 (BERT Hybrid) | 4 dataset tổng hợp |
| `prepare_audio_data.py` | Model 2 — bản cũ (có emotion) | `spotify_dataset.csv` |
| `prepare_tracks_data.py` | Model 2 — bản mới (1.2M bài) | `tracks_features.csv` |

---

## 📄 File 1: `prepare_data.py` — Preprocessing Model 1

### Cấu hình

```python
class DataPrepConfig:
    DATASET_DIR = PROJECT_ROOT / "dataset"
    PRIMARY_DATASET   = "final_milliondataset_BERT_500K_revised.json"  # 551K bài
    SECONDARY_DATASET = "900k Definitive Spotify Dataset.json"          # 900K bài
    SPOTIFY_CSV       = "spotify_dataset.csv"
    TRACKS_FEATURES   = "tracks_features.csv"                          # 1.2M bài

    BERT_MODEL   = "bert-base-uncased"
    MAX_LENGTH   = 256
    EMBEDDING_DIM = 768      # BERT base có 768 chiều

    TRAIN_RATIO = 0.8
    VAL_RATIO   = 0.1
    TEST_RATIO  = 0.1

    AUDIO_FEATURES = ['energy', 'danceability', 'valence', 'tempo',
                      'acousticness', 'instrumentalness', 'speechiness',
                      'liveness', 'loudness', 'key', 'mode']
```

### Bước 1 — Chuẩn hóa tên cột (`normalize_columns`)

```python
COLUMN_MAPPING = {
    'Artist(s)':    'artist',
    'song':         'song_name',
    'Positiveness': 'valence',       # Positiveness = Valence theo Spotify
    'Loudness (db)':'loudness',
    'Energy':       'energy',
}

def normalize_columns(df):
    rename_map = {old: new for old, new in COLUMN_MAPPING.items()
                  if old in df.columns and new not in df.columns}
    df = df.rename(columns=rename_map)

    # Ép kiểu các cột âm thanh sang số (có thể đang là chuỗi)
    for feat in DataPrepConfig.AUDIO_FEATURES:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
    return df
```

### Bước 2 — Chuẩn hóa thang đo 0–100 về 0–1 (`normalize_to_spotify_api`)

```python
def normalize_to_spotify_api(df):
    scale_features = ['energy', 'danceability', 'valence', 'acousticness',
                      'instrumentalness', 'speechiness', 'liveness']

    for feat in scale_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
        if df[feat].max() > 1:      # max > 1 → đang dùng thang 0-100
            df[feat] = df[feat] / 100.0

    # Loudness có thể là chuỗi "−5.3db" → tách bỏ đơn vị "db"
    if df['loudness'].dtype == object:
        df['loudness'] = df['loudness'].str.replace('db', '', case=False)
    df['loudness'] = pd.to_numeric(df['loudness'], errors='coerce')

    # Key: đảm bảo nằm trong 0–11
    df['key'] = pd.to_numeric(df['key'], errors='coerce').fillna(0).astype(int) % 12
    return df
```

### Bước 3 — Gộp 4 dataset thông minh (`load_and_merge_all_datasets`)

```python
def load_and_merge_all_datasets(max_rows=None):
    # Bắt đầu với PRIMARY (chất lượng cao nhất)
    merged_df = primary_df.copy()

    # Tạo "khóa gộp" để nhận diện bài trùng lặp
    merged_df['_merge_key'] = (
        merged_df['artist'].fillna('').str.lower().str.strip() + '|' +
        merged_df['song_name'].fillna('').str.lower().str.strip()
    )
    existing_keys = set(merged_df['_merge_key'].unique())

    # Thêm bài từ 900K — chỉ những bài CHƯA có trong Primary
    unique_secondary = secondary_df[~secondary_df['_merge_key'].isin(existing_keys)]
    merged_df = pd.concat([merged_df[common_cols], unique_secondary[common_cols]])

    # Điền thêm đặc trưng âm thanh thiếu từ tracks_features.csv
    for idx in merged_df[missing_mask].index:
        key = merged_df.loc[idx, '_merge_key']
        if key in tracks_lookup:
            merged_df.loc[idx, feat] = tracks_lookup[key][feat]

    return merged_df
```

### Bước 4 — Xử lý giá trị thiếu (`handle_missing_values`)

```python
def handle_missing_values(df):
    # Xóa bài không có lời hoặc lời quá ngắn (BERT không xử lý được)
    df = df[df['text'].notna() & (df['text'].str.len() > 50)]

    # Điền âm thanh thiếu bằng MEDIAN (ít bị ảnh hưởng bởi outlier hơn mean)
    for feat in DataPrepConfig.AUDIO_FEATURES:
        if feat in df.columns and df[feat].isnull().sum() > 0:
            df[feat] = df[feat].fillna(df[feat].median())

    df['emotion'] = df['emotion'].fillna('neutral')
    df['genre']   = df['genre'].fillna('Unknown')
    return df
```

### Bước 5 — Chia train/val/test (`create_splits`)

```python
def create_splits(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Tách 10% làm test
    train_val, test = train_test_split(df, test_size=test_ratio, random_state=42)

    # Từ phần còn lại, tách val (= 10% / 90% ≈ 11.1%)
    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_ratio_adj, random_state=42)

    return train, val, test   # 80% | 10% | 10%
```

### Bước 6 — Làm sạch lời (`clean_lyrics`)

```python
def clean_lyrics(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Xóa URL
    text = re.sub(r'<[^>]+>', '', text)                  # Xóa HTML
    text = re.sub(r'\[.*?\]', '', text)                  # Xóa [Verse 1], [Chorus]
    text = text.lower()
    text = re.sub(r'[^\w\s\'\"\.\\,\!\?]', ' ', text)   # Giữ chữ, xóa ký tự lạ
    return re.sub(r'\s+', ' ', text).strip()
```

### Bước 7 — Tạo BERT Embeddings (`create_bert_embeddings`)

```python
class LyricsDataset(Dataset):
    """Wrapper để đưa lời hàng loạt qua BERT."""
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(text, max_length=256, padding='max_length',
                                 truncation=True, return_tensors='pt')
        return {'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)}

def create_bert_embeddings(df, output_path, batch_size=32, device="cuda"):
    cleaned_lyrics = [clean_lyrics(t) for t in df['text'].tolist()]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertModel.from_pretrained("bert-base-uncased")
    model.to(device).eval()

    all_embeddings = []
    with torch.no_grad():
        for batch in DataLoader(LyricsDataset(cleaned_lyrics, tokenizer), batch_size=32):
            outputs = model(input_ids=batch['input_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device))
            # [CLS] token (vị trí 0) = đại diện toàn bộ bài hát
            cls_emb = outputs.last_hidden_state[:, 0, :]  # shape: [B, 768]
            all_embeddings.append(cls_emb.cpu().numpy())

    embeddings = np.vstack(all_embeddings)   # shape: (N_songs, 768)
    np.save(output_path, embeddings)         # Lưu để dùng khi training
```

---

## 📄 File 2: `prepare_audio_data.py` — Preprocessing Model 2 (có emotion)

```python
# Tải spotify_dataset.csv (551K bài, có emotion label)
df = pd.read_csv('dataset/spotify_dataset.csv')
df = df.rename(columns={'Artist(s)': 'artist', 'Positiveness': 'valence', ...})

# Chuẩn hóa 0-100 → 0-1
for col in ['energy', 'danceability', 'valence', ...]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].max() > 1:
        df[col] = df[col] / 100.0

# Encode Key (tên nốt nhạc → số)
KEY_MAP = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
           'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
df['key'] = df['key'].apply(lambda k: KEY_MAP.get(str(k).split()[0], 5) / 11.0)

# Encode Emotion (chữ → số)
EMOTION_MAP = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'love': 4, 'surprise': 5}
df['emotion_label'] = df['emotion'].str.lower().map(EMOTION_MAP)

# Chia 80/10/10, stratified theo emotion (đảm bảo tỉ lệ cảm xúc đều)
train, temp = train_test_split(df, test_size=0.2, stratify=df['emotion_label'], random_state=42)
val, test   = train_test_split(temp, test_size=0.5, stratify=temp['emotion_label'], random_state=42)

# Lưu CSV
train.to_csv('data/processed/audio_train.csv', index=False)
val.to_csv('data/processed/audio_val.csv',   index=False)
test.to_csv('data/processed/audio_test.csv', index=False)

# Lưu stats để normalize khi inference
stats = {'mean': train[features].mean().tolist(), 'std': train[features].std().tolist()}
json.dump(stats, open('models/audio2_stats.json', 'w'))
```

---

## 📄 File 3: `prepare_tracks_data.py` — Preprocessing Model 2 (1.2M bài)

```python
# Tải tracks_features.csv (1.2M bài, KHÔNG có emotion)
df = pd.read_csv('dataset/tracks_features.csv')
df = df.rename(columns={'name': 'song_name', 'artists': 'artist'})

# Chuẩn hóa (tương tự nhưng buffer 1.5 thay vì 1.0)
for col in SCALE_100:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].max() > 1.5:   # buffer rộng hơn
        df[col] = df[col] / 100.0

# Key: số nguyên 0-11 → chia 11 → 0-1
df['key'] = pd.to_numeric(df['key'], errors='coerce') / 11.0

# Chia 90/5/5 (dataset lớn hơn nên test nhỏ hơn cũng đủ)
train, temp = train_test_split(df, test_size=0.10, random_state=42)  # không stratify!
val, test   = train_test_split(temp, test_size=0.5, random_state=42)

def save_split(split_df, split_name):
    # Lưu features dạng numpy (nhẹ hơn CSV)
    features = split_df[AUDIO_COLS_RAW].values.astype(np.float32)
    np.save(f'data/processed/tracks/features_{split_name}.npy', features)

    # Lưu metadata dạng parquet (nhanh hơn CSV, nén tốt hơn)
    meta_cols = ['id', 'song_name', 'artist', 'year', 'album']
    split_df[[c for c in meta_cols if c in split_df.columns]].to_parquet(
        f'data/processed/tracks/meta_{split_name}.parquet', index=False)

save_split(train, 'train')
save_split(val,   'val')
save_split(test,  'test')

# Stats để Z-score normalize khi inference
stats = {'mean': train[AUDIO_COLS_RAW].mean().tolist(),
         'std':  train[AUDIO_COLS_RAW].std().tolist()}
json.dump(stats, open('models/tracks_stats.json', 'w'))
```

---

## 📊 So sánh 3 file

| | `prepare_data.py` | `prepare_audio_data.py` | `prepare_tracks_data.py` |
|---|---|---|---|
| **Dataset** | 4 nguồn gộp lại | spotify_dataset.csv (551K) | tracks_features.csv (1.2M) |
| **Có lời bài hát?** | ✅ Có | ❌ Không | ❌ Không |
| **Có emotion?** | ✅ Có | ✅ Có | ❌ Không |
| **BERT Embedding** | ✅ Tạo ra | ❌ Không | ❌ Không |
| **Encode Key** | Số 0-11 → %12 | Tên nốt (C, D...) | Số 0-11 → /11 |
| **Chia dữ liệu** | 80/10/10 | 80/10/10 stratified | 90/5/5 |
| **Output** | .csv + .npy BERT | .csv | .npy + .parquet |
| **Stats file** | Không lưu riêng | `audio2_stats.json` | `tracks_stats.json` |
