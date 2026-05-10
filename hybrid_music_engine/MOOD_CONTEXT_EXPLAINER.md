# 🎵 Mood & Context Recommendation — Giải thích chi tiết

Tài liệu này giải thích đầy đủ cách 2 chức năng **"By Mood"** và **"By Context"** hoạt động, từ lúc người dùng bấm nút cho đến khi nhận được danh sách bài hát.

Hệ thống gồm **3 lớp (layer)** liên kết với nhau:

```
Layer 1 — Frontend     : HTML buttons + JavaScript (main.js)
Layer 2 — Backend API  : FastAPI endpoints (app.py)
Layer 3 — Engine lõi   : Thuật toán lọc bài hát (inference.py)
```

---

## Layer 1 — Frontend: Nút bấm và hàm JavaScript

### 1.1 HTML — Nút bấm kích hoạt (`index.html`)

```html
<!-- ============================================================
     Khu vực "By Mood" — hiển thị khi người dùng chọn tab mood
     ============================================================ -->
<div class="filter-container hidden" id="mood-mode">
    <h3 class="filter-title">How are you feeling?</h3>
    <div class="filter-buttons">

        <!-- Mỗi nút gọi hàm searchByMood() với tên mood làm tham số -->
        <button class="mood-btn" data-mood="Happy" onclick="searchByMood('Happy')">
            <span class="mood-emoji">😊</span>
            <span>Happy</span>
        </button>
        <button class="mood-btn" data-mood="Sad" onclick="searchByMood('Sad')">
            <span class="mood-emoji">😢</span>
            <span>Sad</span>
        </button>
        <button class="mood-btn" data-mood="Energetic" onclick="searchByMood('Energetic')">
            <span class="mood-emoji">⚡</span>
            <span>Energetic</span>
        </button>
        <button class="mood-btn" data-mood="Calm" onclick="searchByMood('Calm')">
            <span class="mood-emoji">😌</span>
            <span>Calm</span>
        </button>
        <button class="mood-btn" data-mood="Angry" onclick="searchByMood('Angry')">
            <span class="mood-emoji">😠</span>
            <span>Angry</span>
        </button>
    </div>
</div>

<!-- ============================================================
     Khu vực "By Context" — hiển thị khi người dùng chọn tab context
     ============================================================ -->
<div class="filter-container hidden" id="context-mode">
    <h3 class="filter-title">What are you doing?</h3>
    <div class="filter-buttons">

        <!-- Mỗi nút gọi hàm searchByContext() với tên context làm tham số -->
        <button class="context-btn" data-context="Party" onclick="searchByContext('Party')">
            <span class="context-emoji">🎉</span>
            <span>Party</span>
        </button>
        <button class="context-btn" data-context="Workout" onclick="searchByContext('Workout')">
            <span class="context-emoji">💪</span>
            <span>Workout</span>
        </button>
        <button class="context-btn" data-context="Study" onclick="searchByContext('Study')">
            <span class="context-emoji">📚</span>
            <span>Study</span>
        </button>
        <button class="context-btn" data-context="Relax" onclick="searchByContext('Relax')">
            <span class="context-emoji">🌸</span>
            <span>Relax</span>
        </button>
        <button class="context-btn" data-context="Driving" onclick="searchByContext('Driving')">
            <span class="context-emoji">🚗</span>
            <span>Driving</span>
        </button>
    </div>
</div>
```

> **Điểm chú ý:** Mỗi nút dùng thuộc tính `onclick` gọi trực tiếp hàm JavaScript,
> truyền tên nhãn (VD: `'Happy'`, `'Party'`) dưới dạng chuỗi ký tự.

---

### 1.2 JavaScript — Hàm gọi API (`main.js`)

```javascript
// ============================================================
// Hàm By Mood — được gọi khi user bấm nút mood
// Ví dụ: searchByMood('Happy')
// ============================================================
async function searchByMood(mood) {

    showLoading();  // Hiện spinner loading trên giao diện

    try {
        // Gọi HTTP GET đến backend, truyền mood vào URL path
        // Ví dụ: GET /api/mood/Happy?limit=10
        const response = await fetch(`/api/mood/${mood}?limit=10`);

        // Parse JSON từ response body
        const data = await response.json();

        // Nếu server trả về lỗi (HTTP 4xx/5xx)
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get recommendations');
        }

        // data là mảng 10 bài hát [{song, artist, genre, emotion}, ...]
        // displayResults() render chúng thành card trên giao diện
        displayResults(data, `${mood} Mood`, data.length);
        // Ví dụ title: "Happy Mood"

    } catch (error) {
        console.error('Mood search error:', error);
        showToast(error.message, 'error');  // Hiện thông báo lỗi
        hideResults();
    } finally {
        hideLoading();  // Tắt spinner dù thành công hay lỗi
    }
}


// ============================================================
// Hàm By Context — được gọi khi user bấm nút context
// Ví dụ: searchByContext('Party')
// ============================================================
async function searchByContext(context) {

    showLoading();

    try {
        // Gọi HTTP GET đến backend, truyền context vào URL path
        // Ví dụ: GET /api/context/Party?limit=10
        const response = await fetch(`/api/context/${context}?limit=10`);

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get recommendations');
        }

        // Hiển thị kết quả với label "<Context> Vibes"
        displayResults(data, `${context} Vibes`, data.length);
        // Ví dụ title: "Party Vibes"

    } catch (error) {
        console.error('Context search error:', error);
        showToast(error.message, 'error');
        hideResults();
    } finally {
        hideLoading();
    }
}
```

> **Sự khác biệt duy nhất** giữa 2 hàm:
> - URL endpoint: `/api/mood/` vs `/api/context/`
> - Label hiển thị: `"Happy Mood"` vs `"Party Vibes"`
>
> Logic xử lý còn lại **hoàn toàn giống nhau**.

---

## Layer 2 — Backend API: FastAPI Endpoint (`app.py`)

```python
# ============================================================
# Endpoint By Mood
# URL: GET /api/mood/{mood}
# Ví dụ: GET /api/mood/Happy?limit=10
# ============================================================

@app.get("/api/mood/{mood}", response_model=List[SongResult])
async def get_by_mood(
    mood: str,                                    # Lấy từ URL path (VD: "Happy")
    limit: int = Query(10, ge=1, le=50)           # Query param tuỳ chọn, mặc định 10
):
    # Kiểm tra engine recommendation đã được khởi động chưa
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not initialized"
        )

    # Validation: từ chối mood không hợp lệ
    valid_moods = ["happy", "sad", "energetic", "calm", "angry"]
    if mood.lower() not in valid_moods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mood. Available: {', '.join(valid_moods)}"
        )

    # Gọi hàm engine để tính toán → nhận về List[Dict]
    results = engine.get_recommendations_by_mood(mood, top_k=limit)

    # Chuyển mỗi dict thành Pydantic model SongResult → FastAPI tự serialize thành JSON
    return [SongResult(**r) for r in results] if results else []


# ============================================================
# Endpoint By Context
# URL: GET /api/context/{context}
# Ví dụ: GET /api/context/Party?limit=10
# ============================================================

@app.get("/api/context/{context}", response_model=List[SongResult])
async def get_by_context(
    context: str,                                 # Lấy từ URL path (VD: "Party")
    limit: int = Query(10, ge=1, le=50)
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")

    # Validation: từ chối context không hợp lệ
    valid_contexts = ["party", "workout", "study", "relax", "driving"]
    if context.lower() not in valid_contexts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid context. Available: {', '.join(valid_contexts)}"
        )

    # Context endpoint là wrapper mỏng — uỷ thác xử lý cho engine
    results = engine.get_recommendations_by_context(context, top_k=limit)
    return [SongResult(**r) for r in results] if results else []
```

> **Thiết kế tốt:** Cả 2 endpoint đều:
> 1. **Validate** input trước khi xử lý (tránh crash engine)
> 2. **Uỷ thác** toàn bộ logic nghiệp vụ cho `engine` — API layer chỉ lo routing
> 3. **Trả về List rỗng** thay vì lỗi 500 khi không có kết quả

---

## Layer 3 — Engine lõi: Thuật toán lọc bài hát (`inference.py`)

Đây là phần quan trọng nhất — nơi thuật toán thực sự tính toán.

### 3.1 `get_recommendations_by_mood()` — Bộ lọc theo âm nhạc

```python
def get_recommendations_by_mood(self, mood: str, top_k: int = 10) -> List[Dict]:

    # Kiểm tra dataset đã được load chưa
    if self.song_data is None:
        return []

    # ── Bước 1: Tra bảng ánh xạ Mood → điều kiện lọc ──────────────────────
    #
    # Mỗi mood ánh xạ vào bộ 3 điều kiện:
    #
    # emotion : Nhãn cảm xúc lời bài hát trong dataset
    #           (được phân tích bởi NLP model khi preprocessing)
    #           Các giá trị: 'joy', 'sadness', 'anger', 'fear', 'love', 'surprise'
    #
    # valence : Chỉ số "tích cực" của âm nhạc (nguồn: Spotify Audio Features)
    #           0.0 = cực kỳ tiêu cực/u ám
    #           1.0 = cực kỳ vui tươi/tích cực
    #
    # energy  : Chỉ số "năng lượng" của âm nhạc (nguồn: Spotify Audio Features)
    #           0.0 = nhẹ nhàng, yên tĩnh
    #           1.0 = sôi động, mạnh mẽ
    #
    mood_mapping = {
        'happy':     {
            'emotion': ['joy'],       # Lời bài hát thể hiện niềm vui
            'valence': (0.6, 1.0),    # Âm nhạc tươi sáng
            'energy':  (0.5, 1.0)     # Năng lượng trung bình đến cao
        },
        'sad':       {
            'emotion': ['sadness'],   # Lời bài hát buồn bã
            'valence': (0.0, 0.4),    # Âm nhạc u ám
            'energy':  (0.0, 0.5)     # Năng lượng thấp
        },
        'energetic': {
            'emotion': ['joy'],       # Lời vui, phấn khích
            'valence': (0.5, 1.0),    # Âm nhạc tích cực
            'energy':  (0.7, 1.0)     # Năng lượng rất cao
        },
        'calm':      {
            'emotion': [],            # Không lọc theo emotion (quá hạn hẹp)
            'valence': (0.4, 0.7),    # Âm nhạc trung tính đến nhẹ nhàng
            'energy':  (0.0, 0.4)     # Năng lượng thấp
        },
        'angry':     {
            'emotion': ['anger'],     # Lời bài hát tức giận
            'valence': (0.0, 0.4),    # Âm nhạc tối tăm
            'energy':  (0.7, 1.0)     # Năng lượng rất cao (intense)
        },
    }

    mood_lower = mood.lower()
    if mood_lower not in mood_mapping:
        mood_lower = 'happy'                    # Fallback an toàn
    filters = mood_mapping[mood_lower]          # Lấy bộ điều kiện


    # ── Bước 2: Lọc DataFrame theo 3 điều kiện ─────────────────────────────

    filtered = self.song_data.copy()            # Copy để giữ nguyên data gốc

    # Lọc theo emotion
    if filters['emotion'] and 'emotion' in filtered.columns:
        filtered = filtered[filtered['emotion'].isin(filters['emotion'])]
        # isin(['joy']) → giữ lại hàng có emotion == 'joy'
        # isin([]) bỏ qua → không lọc emotion (dùng cho mood 'calm')

    # Lọc theo valence (khoảng [min, max])
    if 'valence' in filtered.columns:
        filtered = filtered[
            (filtered['valence'] >= filters['valence'][0]) &   # >= min
            (filtered['valence'] <= filters['valence'][1])     # <= max
        ]

    # Lọc theo energy (khoảng [min, max])
    if 'energy' in filtered.columns:
        filtered = filtered[
            (filtered['energy'] >= filters['energy'][0]) &
            (filtered['energy'] <= filters['energy'][1])
        ]


    # ── Bước 3: Sắp xếp theo độ phổ biến ───────────────────────────────────

    if 'Popularity' in filtered.columns:
        filtered = filtered.sort_values('Popularity', ascending=False)
        # ascending=False → bài phổ biến nhất (điểm cao nhất) lên đầu


    # ── Bước 4: Random Sampling — tránh kết quả trùng lặp ─────────────────
    #
    # VẤN ĐỀ CŨ (trước khi fix):
    #   Luôn lấy Top 10 → mỗi lần bấm "Happy" đều ra đúng 10 bài đó
    #
    # CÁCH FIX:
    #   Lấy pool 100 bài phổ biến nhất → sample ngẫu nhiên 10
    #   → Kết quả khác nhau mỗi lần, tạo sự đa dạng cho người dùng
    #
    if len(filtered) > top_k:
        pool_size = min(100, len(filtered))          # Lấy tối đa 100 ứng viên
        filtered = filtered.head(pool_size).sample(n=top_k)  # Random 10 từ pool

        # Sắp xếp lại 10 bài được chọn theo popularity (tùy chọn)
        if 'Popularity' in filtered.columns:
            filtered = filtered.sort_values('Popularity', ascending=False)


    # ── Bước 5: Đóng gói kết quả thành List[Dict] ──────────────────────────

    song_col, artist_col = self._get_column_names()   # Tự động detect tên cột
    recommendations = []

    for _, row in filtered.head(top_k).iterrows():
        recommendations.append({
            'song':    row.get(song_col, ''),
            'artist':  row.get(artist_col, ''),
            'genre':   row.get('Genre', row.get('genre', '')),
            'emotion': row.get('emotion', ''),
        })

    return recommendations   # List[Dict] → app.py nhận và serialize thành JSON
```

---

### 3.2 `get_recommendations_by_context()` — Alias của Mood

```python
def get_recommendations_by_context(self, context: str, top_k: int = 10) -> List[Dict]:

    if self.song_data is None:
        return []

    # ── Context chỉ là "alias" của mood ────────────────────────────────────
    #
    # Context = hoạt động đang làm → ánh xạ sang mood phù hợp với hoạt động đó
    #
    # Party   → energetic  (tiệc tùng cần nhạc sôi động năng lượng cao)
    # Workout → energetic  (tập gym cần nhạc hype)
    # Study   → calm       (học tập cần nhạc nhẹ nhàng, tập trung)
    # Relax   → calm       (thư giãn cần nhạc êm dịu)
    # Driving → happy      (lái xe cần nhạc vui tươi, không quá chậm)
    #
    context_mapping = {
        'party':   'energetic',
        'workout': 'energetic',
        'study':   'calm',
        'relax':   'calm',
        'driving': 'happy',
    }

    # Tra bảng → lấy mood tương ứng (fallback: 'happy')
    mood = context_mapping.get(context.lower(), 'happy')

    # TÁI SỬ DỤNG hoàn toàn hàm By Mood — không viết lại logic
    return self.get_recommendations_by_mood(mood, top_k)
```

> **Điểm thiết kế quan trọng:** `get_recommendations_by_context()` **không có logic riêng**.
> Nó chỉ là bảng tra ánh xạ `context → mood`, rồi gọi lại `get_recommendations_by_mood()`.
> Đây là nguyên tắc **DRY (Don't Repeat Yourself)** — tránh viết cùng một logic 2 lần.

---

## Sơ đồ luồng dữ liệu tổng hợp

### Luồng By Mood

```
Người dùng bấm "😊 Happy"
        │
        ▼ (HTML onclick)
searchByMood('Happy')                         [main.js - Layer 1]
        │
        ▼ HTTP GET
/api/mood/Happy?limit=10                      [app.py - Layer 2]
        │ validate: 'happy' ∈ valid_moods ✓
        ▼
engine.get_recommendations_by_mood('Happy', 10)   [inference.py - Layer 3]
        │
        ├─ 1. Tra bảng: Happy → {emotion:['joy'], valence:(0.6,1.0), energy:(0.5,1.0)}
        │
        ├─ 2. Lọc DataFrame:
        │       emotion == 'joy'             → còn ~3000 bài
        │       AND valence ∈ [0.6, 1.0]    → còn ~1500 bài
        │       AND energy ∈ [0.5, 1.0]     → còn ~800 bài
        │
        ├─ 3. Sort: Popularity DESC
        │
        ├─ 4. Lấy Top 100 → sample random 10
        │
        └─ 5. Return [{song, artist, genre, emotion} × 10]
        │
        ▼ JSON response
[{song, artist, genre, emotion}, ...]         [app.py serialize]
        │
        ▼
displayResults(data, 'Happy Mood', 10)        [main.js render UI]
        │
        ▼
10 song cards hiển thị + album art từ Spotify
```

### Luồng By Context

```
Người dùng bấm "🎉 Party"
        │
        ▼
searchByContext('Party')                      [main.js]
        │
        ▼ HTTP GET
/api/context/Party?limit=10                   [app.py]
        │ validate: 'party' ∈ valid_contexts ✓
        ▼
engine.get_recommendations_by_context('Party', 10)   [inference.py]
        │
        ▼ (context_mapping lookup)
'party' → 'energetic'
        │
        ▼ (delegate hoàn toàn)
engine.get_recommendations_by_mood('energetic', 10)
        │
        └─ (giống hệt luồng By Mood với mood = 'energetic')
```

---

## Bảng tóm tắt cho giảng viên

| Câu hỏi | Trả lời |
|---|---|
| **By Mood hoạt động thế nào?** | Lọc DataFrame theo 3 tiêu chí: `emotion + valence + energy` từ bảng ánh xạ cố định |
| **By Context khác By Mood thế nào?** | Context chỉ là **alias của mood** — ánh xạ qua bảng rồi gọi lại hàm By Mood, không có logic riêng |
| **Sao kết quả không trùng mỗi lần?** | Random Sampling: lọc Top 100 phổ biến → `sample(n=10)` ngẫu nhiên → kết quả khác nhau mỗi lần |
| **valence và energy lấy từ đâu?** | Lấy từ **Spotify Audio Features API** trong bước preprocessing dữ liệu |
| **emotion lấy từ đâu?** | Phân tích lời bài hát bằng **NLP model** (GoEmotions/DistilBERT) trong bước preprocessing |
| **Architecture pattern sử dụng?** | 3-layer: HTML (trigger) → FastAPI (validate + route) → Engine (business logic) |
| **Nguyên tắc thiết kế nổi bật?** | **DRY**: Context không viết lại logic, chỉ delegate sang Mood. **Separation of Concerns**: mỗi layer có nhiệm vụ riêng |

---

## Các file liên quan

| File | Vai trò |
|---|---|
| `web_app/templates/index.html` | Định nghĩa nút bấm và gọi JS function |
| `web_app/static/js/main.js` | `searchByMood()`, `searchByContext()`, `displayResults()` |
| `web_app/app.py` | FastAPI endpoint `/api/mood/{mood}` và `/api/context/{context}` |
| `hybrid_music_engine/inference.py` | `get_recommendations_by_mood()`, `get_recommendations_by_context()` |
| `preprocessing/` | Tạo ra cột `emotion`, `valence`, `energy` trong dataset |
