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

---

## 4. Code chi tiết: `image_processor.py`

Dưới đây là toàn bộ source code của module xử lý ảnh, kèm giải thích chi tiết từng phần:

### 4.1 Import và kiểm tra CLIP availability

```python
import io
from typing import Tuple, Optional, Dict, List
from PIL import Image  # Thư viện xử lý ảnh (mở file, convert format)

# =============================================================================
# Kiểm tra xem CLIP có thể dùng được không (transformers + torch phải được cài)
# =============================================================================
CLIP_AVAILABLE = False  # Mặc định là False — an toàn khi import thất bại
try:
    from transformers import CLIPProcessor, CLIPModel  # Model CLIP từ HuggingFace
    import torch                                        # PyTorch để chạy model
    CLIP_AVAILABLE = True  # Nếu import thành công → CLIP sẵn sàng
except ImportError as e:
    # Nếu thư viện chưa cài → in cảnh báo nhưng KHÔNG crash app
    # App vẫn chạy được, chỉ tắt tính năng gợi ý từ ảnh
    print(f"[WARNING] CLIP dependencies not available: {e}")
    print("[WARNING] Image-based recommendation feature will be disabled.")
```

**Tại sao dùng `try/except` ở mức module?**
Vì khi deploy lên server (Hugging Face, Docker...), người dùng có thể không cài `torch` (rất nặng ~1GB). Bằng cách bắt lỗi ngay khi import, app vẫn hoạt động bình thường với các tính năng khác — chỉ tắt riêng tính năng phân tích ảnh.

---

### 4.2 Khai báo class `ImageMoodClassifier`

```python
class ImageMoodClassifier:
    """
    Phân loại ảnh thành các nhãn mood/context bằng model CLIP.
    
    Class này đóng gói toàn bộ logic liên quan đến CLIP, tách biệt với
    engine gợi ý chính để code sạch và dễ bảo trì.
    """
    
    # =========================================================================
    # VISUAL_PROMPTS: Bộ từ điển ánh xạ "mô tả ảnh" → (nhãn hệ thống, loại)
    # =========================================================================
    # Đây là "bộ não" của zero-shot classification.
    # CLIP sẽ so sánh ảnh đầu vào với TẤT CẢ các chuỗi text này,
    # rồi chọn ra chuỗi nào "gần" với ảnh nhất.
    
    VISUAL_PROMPTS: Dict[str, Tuple[str, str]] = {
        # --- Nhóm MOOD (cảm xúc) → dùng API get_recommendations_by_mood ---
        
        "a photo of a happy smiling person": ("Happy", "mood"),
        # Ảnh người cười → Happy mood
        
        "a photo of a joyful celebration": ("Happy", "mood"),
        # Ảnh tiệc tùng vui vẻ → cũng Happy (2 prompt → 1 label để tăng độ nhạy)
        
        "a photo of a sad or crying person": ("Sad", "mood"),
        # Ảnh người buồn/khóc → Sad
        
        "a photo of a melancholic scene": ("Sad", "mood"),
        # Ảnh cảnh u ám, cô đơn → Sad
        
        "a photo of an energetic concert or festival": ("Energetic", "mood"),
        # Ảnh concert sôi động → Energetic
        
        "a photo of people jumping or dancing": ("Energetic", "mood"),
        # Ảnh nhảy múa → Energetic
        
        "a peaceful sunset or sunrise landscape": ("Calm", "mood"),
        # Ảnh hoàng hôn yên bình → Calm
        
        "a serene nature scene with trees or water": ("Calm", "mood"),
        # Ảnh thiên nhiên tĩnh lặng → Calm
        
        "a photo of an angry or frustrated person": ("Angry", "mood"),
        # Ảnh người tức giận → Angry
        
        # --- Nhóm CONTEXT (hoạt động) → dùng API get_recommendations_by_context ---
        
        "a party scene with dancing people": ("Party", "context"),
        "a nightclub or disco with colorful lights": ("Party", "context"),
        # Ảnh tiệc, disco → Party context (nhạc sôi động)
        
        "people working out in a gym": ("Workout", "context"),
        "a person running or jogging outdoors": ("Workout", "context"),
        # Ảnh tập gym, chạy bộ → Workout context (nhạc hype)
        
        "a quiet study room or library": ("Study", "context"),
        "a person reading or working at a desk": ("Study", "context"),
        # Ảnh học tập, đọc sách → Study context (nhạc tập trung)
        
        "a relaxing spa or meditation scene": ("Relax", "context"),
        "a cozy living room or bedroom": ("Relax", "context"),
        # Ảnh thư giãn, phòng ấm cúng → Relax context (nhạc nhẹ nhàng)
        
        "a scenic road trip or highway view": ("Driving", "context"),
        "a car interior or driving scene": ("Driving", "context"),
        # Ảnh lái xe, đường xa → Driving context (nhạc road trip)
    }
```

**Tại sao mỗi nhãn có 2 prompt?**
Để tăng xác suất phân loại đúng. Ví dụ: "Happy" được nhận ra cả khi ảnh là người cười lẫn khi ảnh là tiệc tùng. Nếu chỉ có 1 prompt, ảnh tiệc sẽ bị phân loại nhầm sang "Party" context thay vì "Happy" mood.

---

### 4.3 Constructor `__init__`

```python
def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
    """
    Khởi tạo classifier — KHÔNG tải model ngay.
    
    Args:
        model_name: Tên model trên HuggingFace Hub.
                   "clip-vit-base-patch32" là phiên bản nhẹ nhất (~350MB),
                   cân bằng tốt giữa tốc độ và độ chính xác.
    """
    self.model_name = model_name
    
    # Model và processor sẽ là None cho đến khi load_model() được gọi
    # Dùng Optional để type checker không complain
    self.model: Optional["CLIPModel"] = None
    self.processor: Optional["CLIPProcessor"] = None
    
    # Tự động chọn GPU nếu có, ngược lại dùng CPU
    # torch.cuda.is_available() kiểm tra xem CUDA driver có sẵn không
    self.device = "cuda" if CLIP_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    self.is_loaded = False  # Flag để biết model đã load chưa
    
    # Tách riêng list các text prompts để dùng lại trong analyze_image()
    # Thứ tự list này PHẢI khớp với thứ tự keys trong VISUAL_PROMPTS
    self._text_prompts: List[str] = list(self.VISUAL_PROMPTS.keys())
```

**Tại sao không load model trong `__init__`?**
Model CLIP nặng ~350MB. Nếu load ngay khi khởi tạo class, server sẽ mất 10-30 giây trước khi phục vụ request đầu tiên. Bằng cách "lazy load" (chỉ tải khi cần), server khởi động nhanh và tải model lần đầu khi có người thực sự dùng tính năng này.

---

### 4.4 Method `load_model()`

```python
def load_model(self) -> bool:
    """
    Tải CLIP model và processor từ HuggingFace Hub (hoặc cache local).
    
    Returns:
        True nếu tải thành công, False nếu thất bại.
    """
    # Guard 1: Kiểm tra dependencies trước
    if not CLIP_AVAILABLE:
        print("[ERROR] Cannot load CLIP model: dependencies not installed.")
        print("[ERROR] Please run: pip install transformers pillow torch")
        return False
    
    # Guard 2: Tránh load lại nếu đã load rồi (tiết kiệm RAM)
    if self.is_loaded:
        print("[INFO] CLIP model already loaded.")
        return True
    
    try:
        print(f"[INFO] Loading CLIP model: {self.model_name}")
        print(f"[INFO] This may take a moment on first run (downloading ~350MB)...")
        
        # CLIPProcessor: Xử lý cả ảnh lẫn text thành tensor cho model
        # - Với text: tokenize thành input IDs
        # - Với ảnh: resize về 224x224, normalize pixel values
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        # CLIPModel: Model chính với 2 encoder:
        # - vision_model: Mã hóa ảnh thành embedding 512D
        # - text_model: Mã hóa text thành embedding 512D
        self.model = CLIPModel.from_pretrained(self.model_name)
        
        # Chuyển model sang device phù hợp (GPU/CPU)
        self.model = self.model.to(self.device)
        
        # model.eval(): Tắt dropout và batch normalization (chỉ dùng khi training)
        # Bắt buộc làm bước này để inference cho kết quả ổn định và nhanh hơn
        self.model.eval()
        
        self.is_loaded = True
        print(f"[OK] CLIP model loaded successfully on {self.device.upper()}")
        return True
        
    except Exception as e:
        # Bắt mọi lỗi có thể xảy ra (network error, disk full, v.v.)
        print(f"[ERROR] Failed to load CLIP model: {e}")
        self.is_loaded = False
        return False
```

---

### 4.5 Method `analyze_image()` — Trái tim của hệ thống

```python
def analyze_image(self, image_bytes: bytes) -> Tuple[str, str, float]:
    """
    Phân tích ảnh và trả về nhãn mood/context phù hợp nhất.
    
    Returns:
        (label, type, confidence)
        Ví dụ: ("Happy", "mood", 0.853)
    """
    # ── Bước 0: Đảm bảo model đã được tải ──────────────────────────────────
    if not self.is_loaded:
        if not self.load_model():  # Lazy load lần đầu dùng
            raise RuntimeError("Failed to load CLIP model")
    
    # ── Bước 1: Mở và chuẩn hóa ảnh ────────────────────────────────────────
    try:
        # io.BytesIO: Bọc bytes thành file-like object để PIL đọc được
        image = Image.open(io.BytesIO(image_bytes))
        
        # CLIP chỉ xử lý ảnh RGB (3 kênh màu: R, G, B)
        # Nếu ảnh là RGBA (có kênh alpha/transparency) → bỏ kênh alpha
        # Nếu ảnh là Grayscale (1 kênh) → nhân lên 3 kênh giống nhau
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot process image: {e}")
    
    # ── Bước 2: Chuẩn bị input cho CLIP ────────────────────────────────────
    # CLIPProcessor làm 2 việc cùng lúc:
    # 1. Text: Tokenize các chuỗi prompt thành token IDs (như BERT tokenizer)
    # 2. Image: Resize về 224x224 pixels, normalize về [-1, 1]
    # Kết quả: Dict chứa input_ids, attention_mask, pixel_values
    inputs = self.processor(
        text=self._text_prompts,   # Danh sách 20 text prompts
        images=image,              # 1 ảnh PIL đã convert sang RGB
        return_tensors="pt",       # "pt" = PyTorch tensor (không phải numpy)
        padding=True               # Pad các text ngắn để cùng độ dài
    )
    
    # ── Bước 3: Chuyển tensor sang đúng device ──────────────────────────────
    # Nếu model đang trên GPU, tensor cũng phải trên GPU
    # Dict comprehension: xử lý từng tensor (input_ids, pixel_values, ...) 
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
    # ── Bước 4: Chạy inference (forward pass) ───────────────────────────────
    with torch.no_grad():
        # torch.no_grad(): Tắt tính gradient — tiết kiệm RAM, tăng tốc ~2x
        # Chỉ cần gradient khi training, không cần khi chỉ predict
        
        outputs = self.model(**inputs)
        # outputs chứa:
        # - logits_per_image: [1, 20] — điểm tương đồng ảnh vs từng prompt
        # - logits_per_text:  [20, 1] — transpose của trên
        # - image_embeds:     [1, 512] — embedding 512D của ảnh
        # - text_embeds:      [20, 512] — embedding 512D của từng prompt
        
        # logits_per_image[i][j] = dot product giữa embedding ảnh i và text j
        # Giá trị càng cao → ảnh và text càng "gần nhau" trong không gian embedding
        logits_per_image = outputs.logits_per_image  # Shape: [1, 20]
        
        # Softmax: Chuyển logits thô thành xác suất (tổng = 1.0)
        # dim=1: Tính softmax theo chiều text (chuẩn hóa trên 20 prompts)
        # Ví dụ: logits [2.1, 0.3, -0.5, ...] → probs [0.63, 0.09, 0.04, ...]
        probs = logits_per_image.softmax(dim=1)
        
        # Tìm index của prompt có xác suất cao nhất
        # .item(): Chuyển từ tensor 0-D về Python scalar (số thường)
        best_idx = probs.argmax().item()
        
        # Lấy xác suất (confidence) của prompt được chọn
        # probs[0, best_idx]: batch thứ 0 (chỉ có 1 ảnh), prompt thứ best_idx
        confidence = probs[0, best_idx].item()
    
    # ── Bước 5: Tra cứu nhãn từ dictionary ─────────────────────────────────
    # Lấy chuỗi text prompt có điểm cao nhất (VD: "a photo of a happy smiling person")
    best_prompt = self._text_prompts[best_idx]
    
    # Tra ngược vào VISUAL_PROMPTS để lấy nhãn hệ thống
    # VD: "a photo of a happy smiling person" → ("Happy", "mood")
    label, label_type = self.VISUAL_PROMPTS[best_prompt]
    
    print(f"[INFO] Image analysis result: {label} ({label_type}) - {confidence:.1%}")
    # VD in ra: "[INFO] Image analysis result: Happy (mood) - 85.3%"
    
    return label, label_type, confidence
    # Trả về tuple 3 phần tử để app.py xử lý tiếp:
    # - label: "Happy" → truyền vào get_recommendations_by_mood("Happy")
    # - label_type: "mood" → app.py biết gọi API mood hay context
    # - confidence: 0.853 → hiển thị thanh confidence bar trên UI
```

---

### 4.6 Method `get_all_labels()` và function `is_clip_available()`

```python
def get_all_labels(self) -> Dict[str, List[str]]:
    """
    Liệt kê tất cả nhãn mood và context mà hệ thống hỗ trợ.
    Dùng để: validate input, hiển thị UI dropdown, viết docs.
    
    Returns:
        {
            'moods':    ['Angry', 'Calm', 'Energetic', 'Happy', 'Sad'],
            'contexts': ['Driving', 'Party', 'Relax', 'Study', 'Workout']
        }
    """
    moods = set()     # Dùng set để tự động loại trùng lặp
    contexts = set()  # (vì mỗi nhãn có 2 prompt nhưng chỉ cần 1 lần trong list)
    
    # Duyệt qua tất cả values trong VISUAL_PROMPTS
    # Mỗi value là tuple (label, label_type)
    for label, label_type in self.VISUAL_PROMPTS.values():
        if label_type == "mood":
            moods.add(label)       # VD: thêm "Happy", "Sad", ...
        else:
            contexts.add(label)    # VD: thêm "Party", "Workout", ...
    
    # sorted(): Sắp xếp alphabet để output nhất quán và dễ đọc
    return {
        "moods": sorted(list(moods)),
        "contexts": sorted(list(contexts))
    }


# =============================================================================
# Hàm tiện ích ở mức module — dùng để kiểm tra trước khi khởi tạo class
# =============================================================================

def is_clip_available() -> bool:
    """
    Kiểm tra xem CLIP có thể dùng được không.
    
    Dùng trong app.py để quyết định có khởi tạo ImageMoodClassifier không:
    
        if is_clip_available():
            image_classifier = ImageMoodClassifier()
        else:
            image_classifier = None  # Tắt tính năng phân tích ảnh
    
    Returns:
        True nếu transformers và torch đã được cài đặt.
    """
    return CLIP_AVAILABLE  # Trả về flag đã được set lúc import
```

---

## 5. Luồng dữ liệu đầy đủ (End-to-End Flow)

```
Người dùng upload ảnh (JPEG/PNG/WebP)
        │
        ▼
app.py: POST /api/recommend-from-image
  └── image_bytes = await file.read()
        │
        ▼
image_classifier.analyze_image(image_bytes)
  ├── PIL: mở ảnh, convert → RGB
  ├── CLIPProcessor: ảnh → pixel_values tensor [1, 3, 224, 224]
  │                  20 prompts → input_ids tensor [20, 77]
  ├── CLIPModel forward pass:
  │   ├── vision_encoder(pixel_values)  → image_embed [1, 512]
  │   ├── text_encoder(input_ids×20)   → text_embeds [20, 512]
  │   └── dot_product → logits [1, 20] → softmax → probs [1, 20]
  └── best_prompt = prompts[argmax(probs)]
      label, label_type = VISUAL_PROMPTS[best_prompt]
      return ("Happy", "mood", 0.853)
        │
        ▼
app.py: rẽ nhánh theo label_type
  ├── "mood"    → engine.get_recommendations_by_mood("Happy")
  └── "context" → engine.get_recommendations_by_context("Party")
        │
        ▼
JSON response về frontend:
  {
    "detected_label": "Happy",
    "detected_type": "mood",
    "confidence": 0.853,
    "recommendations": [ {song, artist, genre, ...}, × 10 ]
  }
```
