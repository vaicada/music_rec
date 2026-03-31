# PHÂN TÍCH KIẾN TRÚC DUAL-MODEL CHO MUSIC RECOMMENDATION SYSTEM

**Ngày tạo:** 26/01/2026  
**Mục đích:** Đánh giá khả năng triển khai 2 models riêng biệt (Hybrid + Audio-Only) cho các use case khác nhau

---

## MỤC LỤC

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Use Cases & Routing Logic](#2-use-cases--routing-logic)
3. [Pros & Cons Chi Tiết](#3-pros--cons-chi-tiết)
4. [Performance Benchmarks](#4-performance-benchmarks)
5. [Implementation Strategies](#5-implementation-strategies)
6. [Phased Approach (Recommended)](#6-phased-approach-recommended)
7. [Code Examples](#7-code-examples)
8. [Decision Framework](#8-decision-framework)

---

## 1. TỔNG QUAN KIẾN TRÚC

### Current Architecture (Single Model)

```
┌─────────────────────────────────────────┐
│         HYBRID MODEL                    │
│  ┌──────────┐      ┌──────────┐        │
│  │  BERT    │      │  Audio   │        │
│  │  Branch  │      │  MLP     │        │
│  │  (768D)  │      │  (9→64D) │        │
│  └────┬─────┘      └────┬─────┘        │
│       │                 │               │
│       └────────┬────────┘               │
│                │                         │
│         ┌──────▼──────┐                 │
│         │   Fusion    │                 │
│         │  (320→64D)  │                 │
│         └─────────────┘                 │
│                                         │
│  Size: 117 MB                           │
│  Inference: ~400ms (CPU)                │
└─────────────────────────────────────────┘
```

### Proposed Dual-Model Architecture

```
┌──────────────── USER INPUT ────────────────┐
│                                            │
│  Option A: Song Name + Artist             │
│       ↓                                    │
│  ┌─────────────────────────┐              │
│  │   HYBRID MODEL          │              │
│  │   (Lyrics + Audio)      │              │
│  │   Size: 117 MB          │              │
│  │   Inference: ~400ms     │              │
│  │   Accuracy: 85-90%      │              │
│  └────────┬────────────────┘              │
│           │                                │
│           ↓                                │
│  ┌────────────────────┐                   │
│  │  FAISS Index 1     │                   │
│  │  (Hybrid-based)    │                   │
│  │  Size: ~135 MB     │                   │
│  └────────────────────┘                   │
│                                            │
├────────────────────────────────────────────┤
│                                            │
│  Option B: MP3/MP4 Upload                 │
│       ↓                                    │
│  ┌─────────────────────────┐              │
│  │   AUDIO-ONLY MODEL      │              │
│  │   (Audio features only) │              │
│  │   Size: 6 MB            │              │
│  │   Inference: ~50ms      │              │
│  │   Accuracy: 75-80%      │              │
│  └────────┬────────────────┘              │
│           │                                │
│           ↓                                │
│  ┌────────────────────┐                   │
│  │  FAISS Index 2     │                   │
│  │  (Audio-only)      │                   │
│  │  Size: ~135 MB     │                   │
│  └────────────────────┘                   │
│                                            │
└────────────────────────────────────────────┘

Total Storage: 117 + 6 + 135 + 135 = ~393 MB
Total RAM Usage: Similar (all loaded in memory)
```

---

## 2. USE CASES & ROUTING LOGIC

### Use Case Matrix

| Input Type | Model Used | FAISS Index | Accuracy | Speed |
|-----------|-----------|-------------|----------|-------|
| **Song Name + Artist** | Hybrid | Index 1 | 85-90% | ~1s (with AcousticBrainz) |
| **Song Name + Lyrics** | Hybrid | Index 1 | 85-90% | ~1s |
| **MP3 Upload (unknown)** | Audio-Only | Index 2 | 75-80% | ~11-15s |
| **MP3 Upload + Lyrics** | Audio-Only | Index 2 | 75-80% | ~11-15s |

### Routing Decision Tree

```python
def route_to_model(request):
    """
    Decision logic cho dual-model architecture
    """
    if request.has_song_name or request.has_lyrics:
        # Path A: Use Hybrid Model
        return {
            'model': 'hybrid',
            'index': 'faiss_index_1',
            'expected_accuracy': 0.85,
            'processing_time': '~1s'
        }
    
    elif request.has_audio_file:
        # Path B: Use Audio-Only Model
        return {
            'model': 'audio_only',
            'index': 'faiss_index_2',
            'expected_accuracy': 0.75,
            'processing_time': '~11-15s'
        }
    
    else:
        raise ValueError("Invalid input")
```

---

## 3. PROS & CONS CHI TIẾT

### ✅ ADVANTAGES (Ưu Điểm)

#### A1. Specialized Models = Better Performance

**Hybrid Model (Song Name Search):**

- Trained with full dataset (lyrics + audio)
- Optimized for semantic + audio matching
- Accuracy: **85-90%**
- Best for: Users who know song name

**Audio-Only Model (Upload Search):**

- Trained only on audio features
- No dummy/zero embeddings needed
- Accuracy: **75-80%** (vs 65-70% with hybrid + dummy)
- Best for: Unknown songs, humming, snippets

**Improvement:** +10% accuracy for upload case

---

#### A2. Faster Inference for Upload Case

```python
# Timing breakdown:

# CURRENT (Hybrid with dummy lyrics):
BERT forward pass:      ~300ms ❌ (wasted on zeros)
Audio MLP:             ~50ms
Fusion:                ~50ms
──────────────────────────────
Total:                 ~400ms

# WITH AUDIO-ONLY MODEL:
Audio MLP:             ~50ms ✅
Output layer:          ~1ms
──────────────────────────────
Total:                 ~51ms

Speedup: 7.8x faster (400ms → 51ms)
```

**Real-world impact:**

- Upload workflow: 15s → 11s (giảm 27%)
- Better user experience
- Less CPU load per request

---

#### A3. Smaller Model Size

```
Hybrid Model:
├── BERT weights:      110 MB
├── Audio MLP:         5 MB
├── Fusion layers:     2 MB
└── Total:             117 MB

Audio-Only Model:
├── Audio MLP:         5 MB
├── Output layer:      1 MB
└── Total:             6 MB

Size reduction: 95% smaller!
```

**Benefits:**

- Faster loading (50ms vs 2s)
- Can cache in memory easily
- Better for edge deployment

---

#### A4. Clean Embeddings (No Dummy Data)

**Problem với current approach:**

```python
# When user uploads MP3 without lyrics:
lyrics_embedding = np.zeros(768)  # Dummy

# Model được train với real lyrics
# → Confusion khi nhận zero vector
# → Embeddings bị "polluted"
# → Accuracy giảm
```

**Solution với Audio-Only:**

```python
# Model chỉ train với audio
# → Không expect lyrics
# → Clean, focused embeddings
# → Better accuracy
```

---

#### A5. Independent Optimization

```python
# Có thể improve từng model riêng:

Hybrid Model optimization:
- Better lyrics preprocessing
- Advanced NLP techniques
- Emotion-aware embeddings

Audio-Only optimization:
- Better audio feature extraction
- Advanced signal processing
- Genre-specific tuning

# Không conflict với nhau
```

---

#### A6. Flexible Deployment

```python
# Scenarios:

# Scenario 1: Resource-constrained (mobile)
Deploy only: Audio-Only Model (6 MB)
Use case: Quick audio matching

# Scenario 2: Full-featured (cloud)
Deploy both: Hybrid + Audio-Only
Use case: Complete music search

# Scenario 3: Web app (medium)
Deploy: Hybrid (main)
Lazy-load: Audio-Only (khi cần)
```

---

#### A7. Better Error Handling & Fallbacks

```python
def search_with_robust_fallback(input_data):
    try:
        # Try primary model
        if input_data.type == 'name':
            return hybrid_model.search(input_data)
        else:
            return audio_only_model.search(input_data)
    
    except Exception as e:
        logger.error(f"Primary model failed: {e}")
        
        # Fallback to alternative
        if input_data.has_audio:
            return audio_only_model.search(input_data)
        else:
            # Use hybrid with estimated features
            return hybrid_model.search_with_defaults(input_data)
```

---

#### A8. A/B Testing Capability

```python
# Dễ dàng so sánh performance:

# Test 1: Audio-Only vs Hybrid (with dummy)
for song in test_set:
    result_audio = audio_only_model.search(song.audio)
    result_hybrid = hybrid_model.search(song.audio, dummy_lyrics=True)
    
    compare_accuracy(result_audio, result_hybrid)

# Choose better approach
```

---

### ❌ DISADVANTAGES (Nhược Điểm)

#### D1. Development & Maintenance Overhead

**Training Time:**

```
Hybrid Model:
- Data preparation:    2-3 days
- Training:           10-15 hours
- Hyperparameter:     2-3 days
- Total:              ~5-7 days

Audio-Only Model:
- Data preparation:    1-2 days (easier)
- Training:           5-8 hours (faster)
- Hyperparameter:     1-2 days
- Total:              ~3-5 days

TOTAL: 8-12 days (vs 5-7 for single model)
```

**Maintenance:**

- 2x testing effort
- 2x documentation
- 2x debugging complexity
- 2x deployment pipelines

---

#### D2. Two FAISS Indices Required

**Storage Impact:**

```
FAISS Index 1 (Hybrid embeddings):
- Vectors: 551,443 × 64 dimensions × 4 bytes
- Size: ~135 MB

FAISS Index 2 (Audio-Only embeddings):
- Vectors: 551,443 × 64 dimensions × 4 bytes
- Size: ~135 MB

Total: ~270 MB (vs 135 MB for single)

Index build time: 2x (10-20 minutes total)
```

**Implications:**

- More storage needed
- Longer startup time
- Need to sync both indices

---

#### D3. Inconsistent Results

**Scenario:**

```python
# User A: Search by name
query = "Shape of You - Ed Sheeran"
results_A = hybrid_model.search(query)
# → [Song1, Song2, Song3, Song4, Song5]

# User B: Upload same song as MP3
audio_file = "shape_of_you.mp3"
results_B = audio_only_model.search(audio_file)
# → [Song2, Song4, Song1, Song6, Song7]

# Results order different!
# Similarity scores different!
```

**Problems:**

- User confusion
- Harder to validate
- Different embeddings spaces

---

#### D4. Memory & Resource Usage

**RAM Breakdown:**

```
Component               Single Model    Dual Model
─────────────────────────────────────────────────
Hybrid Model            117 MB          117 MB
Audio-Only Model        -               6 MB
FAISS Index 1           135 MB          135 MB
FAISS Index 2           -               135 MB
During inference        ~500 MB         ~500 MB
─────────────────────────────────────────────────
Peak Total              ~752 MB         ~893 MB (+18%)

On CPU v2 (16 GB):
Max users (single):     ~20 concurrent
Max users (dual):       ~17 concurrent
```

**Cold Start:**

```
Single Model: ~2-3s to load
Dual Model:   ~4-5s to load (slower)
```

---

#### D5. Code Complexity

**Lines of Code Increase:**

```python
# Current (single model):
- Model loading:        ~50 lines
- Inference logic:      ~30 lines
- FAISS integration:    ~40 lines
- Total:                ~120 lines

# Dual model:
- Model loading:        ~100 lines (2x models)
- Routing logic:        ~50 lines (NEW)
- Inference logic:      ~60 lines (branching)
- FAISS integration:    ~80 lines (2x indices)
- Error handling:       ~40 lines (fallbacks)
- Total:                ~330 lines (+175%)

More code = More bugs = More testing
```

---

#### D6. Training Challenges

**Audio-Only Model Training Issues:**

1. **Lower accuracy ceiling**
   - No lyrics signal → missing semantic info
   - Expected: 75-80% (vs 85-90% for hybrid)
   - Gap: ~10% (significant)

2. **Training strategy complexity**

   ```python
   # Option A: Train from scratch
   - Pro: Clean model
   - Con: May underperform
   
   # Option B: Fine-tune from hybrid
   - Pro: Transfer learning
   - Con: Complex implementation
   
   # Option C: Knowledge distillation
   - Pro: Best accuracy potential
   - Con: Requires expertise + time
   ```

3. **Hyperparameter tuning**
   - Different optimal settings than hybrid
   - Need separate tuning process
   - Time consuming

---

#### D7. Deployment Complexity

**Hugging Face Spaces Deployment:**

```yaml
# Single Model (current):
files:
  - models/best_model.pth          (117 MB)
  - models/faiss_index.bin         (135 MB)
  - models/audio_stats.json        (1 KB)

config:
  startup_time: ~3s
  complexity: Low

# Dual Model:
files:
  - models/hybrid_model.pth        (117 MB)
  - models/audio_only_model.pth    (6 MB)
  - models/faiss_index_hybrid.bin  (135 MB)
  - models/faiss_index_audio.bin   (135 MB)
  - models/audio_stats.json        (1 KB)
  - models/model_config.json       (NEW)

config:
  startup_time: ~5s
  complexity: Medium
  failure_points: 2x more
```

---

#### D8. Version Control & Sync

**Challenge:**

```
When updating dataset:
1. Retrain hybrid model      → 10-15 hours
2. Retrain audio-only model  → 5-8 hours
3. Rebuild FAISS index 1     → 5-10 minutes
4. Rebuild FAISS index 2     → 5-10 minutes
5. Test both models          → 2-3 hours
6. Deploy both               → 1-2 hours

Total: ~18-28 hours (vs ~13-18 for single)

Risk: Models out of sync if one fails
```

---

## 4. PERFORMANCE BENCHMARKS

### Accuracy Comparison (Estimated)

| Scenario | Current (Hybrid + Dummy) | Dual Model (Audio-Only) | Gain |
|----------|-------------------------|------------------------|------|
| **Name search (with lyrics)** | 85-90% | 85-90% | 0% |
| **Upload (unknown song)** | 65-70% | 75-80% | **+10%** |
| **Upload (instrumental)** | 60-65% | 75-80% | **+15%** |
| **Upload (poor audio quality)** | 55-60% | 70-75% | **+15%** |

### Speed Comparison

| Stage | Current | Dual Model | Improvement |
|-------|---------|-----------|-------------|
| **Model inference** | 400ms | 51ms | **7.8x faster** |
| **Total upload workflow** | ~15s | ~11s | **27% faster** |
| **Name search workflow** | ~1s | ~1s | Same |

### Resource Usage

| Metric | Single Model | Dual Model | Increase |
|--------|-------------|-----------|----------|
| **Storage** | 252 MB | 393 MB | +56% |
| **RAM (peak)** | 752 MB | 893 MB | +18% |
| **Startup time** | 3s | 5s | +67% |
| **Build time** | 10-15 min | 15-25 min | +50% |

---

## 5. IMPLEMENTATION STRATEGIES

### Strategy 1: Dual Model from Scratch

**Timeline: 2-3 tuần**

```python
Week 1: Train Audio-Only Model
  Day 1-2: Data preparation (audio-only)
  Day 3-5: Model training + tuning
  Day 6-7: Validation + accuracy testing

Week 2: Integration
  Day 8-9: Build FAISS index for audio-only
  Day 10-11: Update backend routing logic
  Day 12-13: Frontend updates
  Day 14: Integration testing

Week 3: Deployment & Testing
  Day 15-16: Deployment to staging
  Day 17-18: A/B testing
  Day 19-20: Bug fixes
  Day 21: Production deployment
```

**Pros:**

- ✅ Immediate accuracy improvement
- ✅ Clean implementation
- ✅ Full features from day 1

**Cons:**

- ❌ Long development time
- ❌ High upfront investment
- ❌ No early user feedback

---

### Strategy 2: Phased Approach (RECOMMENDED)

**Timeline: Ship MVP → Iterate based on data**

#### Phase 1: MVP with Single Model (Week 1)

```python
# Deploy hybrid model with dummy lyrics fallback
def search(request):
    if request.has_name:
        lyrics = fetch_lyrics_or_dummy(request.song, request.artist)
    elif request.has_upload:
        lyrics = np.zeros(768)  # Dummy
    
    return hybrid_model.search(lyrics, audio_features)

# Accuracy: 65-70% for uploads
# Time to market: ~1 week
```

#### Phase 2: Gather Usage Data (Week 2-4)

```python
# Instrument app to track:
- % users using upload vs name search
- Which songs are uploaded most
- User feedback on results quality

# Decision criteria:
if upload_usage > 30% and satisfaction < 70%:
    proceed_to_phase_3()
else:
    optimize_single_model()
```

#### Phase 3: Add Audio-Only Model (Week 5-6)

```python
# Only if data justifies
if should_add_audio_model:
    # Train audio-only model
    # Deploy alongside hybrid
    # Migrate upload feature gradually
```

**Pros:**

- ✅ Fast time-to-market (1 week)
- ✅ Data-driven decision
- ✅ Low risk
- ✅ Easy to rollback

**Cons:**

- ⚠️ Initial accuracy lower (65-70%)
- ⚠️ May need rework later

---

### Strategy 3: Hybrid Approach (Compromise)

**Timeline: 1.5-2 tuần**

```python
# Week 1: Basic audio-only model (không tune kỹ)
- Quick training (3-5 days)
- Basic evaluation
- Deploy beta version

# Week 2: Optimize based on feedback
- Fine-tune if needed
- Fix issues
- Full deployment
```

**Pros:**

- ✅ Faster than full dual model
- ✅ Some accuracy improvement
- ✅ Learn while building

**Cons:**

- ⚠️ May need retraining
- ⚠️ Quality not guaranteed

---

## 6. PHASED APPROACH (RECOMMENDED)

### Why Phased is Best?

1. **Risk Mitigation**: Don't invest 3 weeks if feature isn't used
2. **User Feedback**: Learn what users actually need
3. **Faster Iteration**: Ship → Learn → Improve
4. **Resource Efficient**: Only build what's necessary

### Implementation Code

```python
# app.py - Designed for easy migration

class ModelManager:
    def __init__(self):
        # Load hybrid model (required)
        self.hybrid_model = self.load_model("hybrid_model.pth")
        self.faiss_hybrid = self.load_faiss("faiss_hybrid.bin")
        
        # Audio-only model (optional)
        self.audio_only_model = None
        self.faiss_audio = None
        
        # Try to load audio-only if exists
        if os.path.exists("models/audio_only_model.pth"):
            logger.info("Loading audio-only model...")
            self.audio_only_model = self.load_model("audio_only_model.pth")
            self.faiss_audio = self.load_faiss("faiss_audio.bin")
            logger.info("✅ Dual-model mode enabled")
        else:
            logger.info("⚠️ Running in single-model mode")
    
    def search(self, request):
        """
        Intelligent routing with graceful fallback
        """
        # Route based on input type
        if request.type == "name":
            return self._search_hybrid(request)
        
        elif request.type == "upload":
            # Use audio-only if available
            if self.audio_only_model:
                logger.info("Using audio-only model for upload")
                return self._search_audio_only(request)
            else:
                # Fallback to hybrid with dummy lyrics
                logger.info("Fallback: Using hybrid with dummy lyrics")
                return self._search_hybrid(request, use_dummy=True)
        
        else:
            raise ValueError(f"Unknown request type: {request.type}")
    
    def _search_hybrid(self, request, use_dummy=False):
        if use_dummy:
            lyrics_emb = np.zeros(768)
        else:
            lyrics_emb = self.encode_lyrics(request.lyrics)
        
        audio_features = self.extract_audio(request.audio)
        embedding = self.hybrid_model(lyrics_emb, audio_features)
        
        results = self.faiss_hybrid.search(embedding, k=10)
        return self.format_results(results, source="hybrid")
    
    def _search_audio_only(self, request):
        audio_features = self.extract_audio(request.audio)
        embedding = self.audio_only_model(audio_features)
        
        results = self.faiss_audio.search(embedding, k=10)
        return self.format_results(results, source="audio_only")
    
    def format_results(self, results, source):
        return {
            'results': results,
            'model_used': source,
            'timestamp': datetime.now(),
            'metadata': {
                'model_version': self.get_model_version(source),
                'accuracy_estimate': self.get_accuracy_estimate(source)
            }
        }

# Benefits:
# ✅ Zero code changes needed for Phase 1
# ✅ Drop-in audio-only model for Phase 3
# ✅ Always works (graceful degradation)
# ✅ Easy to A/B test
```

### Migration Path

```python
# Phase 1 Deployment:
$ cp models/hybrid_model.pth /deploy/
$ python app.py  # Runs in single-model mode

# Phase 3 Deployment (sau khi train):
$ cp models/audio_only_model.pth /deploy/
$ cp models/faiss_audio.bin /deploy/
$ python app.py  # Automatically detects and uses dual-model

# No code changes needed!
```

---

## 7. CODE EXAMPLES

### Complete Dual-Model Implementation

```python
# models.py

import torch
import torch.nn as nn

class HybridModel(nn.Module):
    """
    Model hiện tại: BERT + Audio → 64D embedding
    """
    def __init__(self):
        super().__init__()
        
        # BERT branch
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        # Audio branch
        self.audio_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
    
    def forward(self, lyrics_emb, audio_features):
        # Process lyrics
        lyrics_feat = self.bert_proj(lyrics_emb)
        
        # Process audio  
        audio_feat = self.audio_encoder(audio_features)
        
        # Fuse
        combined = torch.cat([lyrics_feat, audio_feat], dim=1)
        embedding = self.fusion(combined)
        
        # L2 normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding


class AudioOnlyModel(nn.Module):
    """
    Model mới: Chỉ Audio → 64D embedding
    """
    def __init__(self):
        super().__init__()
        
        # Audio encoder (deeper than hybrid)
        self.audio_encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64)
        )
    
    def forward(self, audio_features):
        embedding = self.audio_encoder(audio_features)
        
        # L2 normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding


# Training script
def train_audio_only_model():
    """
    Train audio-only model from scratch
    """
    # Load data
    train_df = pd.read_csv("data/processed/train.csv")
    
    # Prepare audio features only
    audio_features = train_df[[
        'energy', 'danceability', 'valence', 'tempo',
        'acousticness', 'instrumentalness', 
        'speechiness', 'liveness', 'key'
    ]].values
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    audio_normalized = scaler.fit_transform(audio_features)
    
    # Labels (emotion for classification task)
    labels = train_df['emotion'].values
    
    # Create dataset
    dataset = AudioDataset(audio_normalized, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Initialize model
    model = AudioOnlyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(50):
        model.train()
        total_loss = 0
        
        for batch_audio, batch_labels in loader:
            optimizer.zero_grad()
            
            # Forward
            embeddings = model(batch_audio)
            
            # Classification head (for training)
            logits = classification_head(embeddings)
            loss = criterion(logits, batch_labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss/len(loader):.4f}")
    
    # Save model
    torch.save(model.state_dict(), "models/audio_only_model.pth")
    
    return model
```

---

## 8. DECISION FRAMEWORK

### Flowchart: Should You Use Dual Model?

```
START
  │
  ▼
┌─────────────────────────────────┐
│ Do you have 2-3 weeks?          │
│ ├─ YES → Continue               │
│ └─ NO → Use Phased Approach     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Is upload a CORE feature?       │
│ ├─ YES → Continue               │
│ └─ NO → Single Model Sufficient │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Can you maintain 2 models?      │
│ ├─ YES → Continue               │
│ └─ NO → Phased Approach         │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Is +10% accuracy worth 2x work? │
│ ├─ YES → USE DUAL MODEL ✅      │
│ └─ NO → Single Model is OK      │
└─────────────────────────────────┘
```

### Quick Decision Matrix

| Your Situation | Recommendation | Rationale |
|---------------|----------------|-----------|
| **MVP / Demo** | Single Model | Fast to market, good enough |
| **Production (low traffic)** | Phased Approach | Start simple, iterate |
| **Production (high traffic)** | Dual Model | Accuracy critical |
| **Research Project** | Dual Model | Explore best architecture |
| **Time-constrained** | Single Model | No time for dual |
| **Resource-rich** | Dual Model | Can afford optimization |

---

## APPENDIX A: Performance Metrics

### Expected Accuracy by Category

| Song Category | Hybrid (w/ dummy) | Audio-Only | Best |
|--------------|------------------|-----------|------|
| Pop songs | 68% | 78% | Audio-Only (+10%) |
| Rock | 65% | 75% | Audio-Only (+10%) |
| Classical | 60% | 72% | Audio-Only (+12%) |
| Electronic | 70% | 80% | Audio-Only (+10%) |
| Jazz | 62% | 70% | Audio-Only (+8%) |
| Hip-hop | 67% | 73% | Audio-Only (+6%) |

### Resource Usage by Model

```yaml
Hybrid Model:
  CPU Usage: 40-50% (2 cores)
  Memory: 700-800 MB peak
  GPU (if available): 20-30%

Audio-Only Model:
  CPU Usage: 10-15% (2 cores)
  Memory: 100-150 MB peak
  GPU (if available): 5-10%

Dual Model (both loaded):
  CPU Usage: 50-65% peak
  Memory: 800-950 MB combined
  Startup: +2s slower
```

---

## APPENDIX B: Training Checklist

### Audio-Only Model Training Checklist

- [ ] **Data Preparation**
  - [ ] Extract audio features from all 551K songs
  - [ ] Normalize features (StandardScaler)
  - [ ] Split train/val/test (80/10/10)
  - [ ] Create DataLoader

- [ ] **Model Architecture**
  - [ ] Design deeper audio encoder (no BERT needed)
  - [ ] Add batch normalization
  - [ ] Tune dropout rates
  - [ ] Define output dimension (64D)

- [ ] **Training Configuration**
  - [ ] Optimizer: AdamW (lr=2e-3)
  - [ ] Scheduler: CosineAnnealingLR
  - [ ] Loss: CrossEntropyLoss + TripletLoss
  - [ ] Epochs: 50 (early stopping)
  - [ ] Batch size: 256

- [ ] **Validation**
  - [ ] Accuracy on val set
  - [ ] Compare with hybrid model
  - [ ] Test on edge cases

- [ ] **FAISS Index**
  - [ ] Generate embeddings for all songs
  - [ ] Build IndexFlatIP
  - [ ] Save index + mappings

- [ ] **Integration**
  - [ ] Update ModelManager
  - [ ] Add routing logic
  - [ ] Test end-to-end

---

## CONCLUSION

### Final Recommendation: **PHASED APPROACH**

**Rationale:**

1. ✅ Ship MVP in 1 week (vs 3 weeks for dual)
2. ✅ Validate upload feature with real users first
3. ✅ Data-driven decision (only add audio-only if needed)
4. ✅ Low risk, high flexibility
5. ✅ Code designed for easy migration

**When to Use Full Dual Model:**

- Upload is confirmed to be heavily used (>30%)
- Accuracy improvement matters to users
- You have resources to maintain 2 models
- Time is not critical

**When to Stick with Single Model:**

- Upload is rarely used (<10%)
- Time to market is critical
- Limited maintenance resources
- Single model accuracy acceptable

---

**Document Version:** 1.0  
**Last Updated:** 26/01/2026  
**Author:** Music Recommender Project Team  
**Status:** For Planning & Reference
