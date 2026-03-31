# Hybrid Music Recommendation Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A modular Python package for music recommendations using a **Hybrid Multi-modal Architecture** combining:

- **BERT** for deep semantic understanding of lyrics
- **Audio Features** from Spotify API
- **Metadata Embeddings** for genre, key, and emotion
- **FAISS** for fast similarity search

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Song Input                  │
                    │   (Lyrics + Audio Features + Metadata)   │
                    └────────────────┬────────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
          ▼                          ▼                          ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Branch A: NLP  │      │ Branch B: Audio │      │Branch C: Metadata│
│                 │      │                 │      │                 │
│  BERT Encoder   │      │  MLP Encoder    │      │Embedding Layers │
│  (Lyrics)       │      │  (11 features)  │      │(Genre, Key,     │
│                 │      │                 │      │ Emotion)        │
│  Output: 256D   │      │  Output: 64D    │      │  Output: 32D    │
└────────┬────────┘      └────────┬────────┘      └────────┬────────┘
          │                       │                        │
          └───────────────────────┼────────────────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │       Fusion Layer          │
                    │  Concatenate (352D)         │
                    │  Dense: 352 → 256 → 128     │
                    │  Output: 64D (L2 Normalized)│
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │        FAISS Index          │
                    │  IndexFlatL2 (Exact Search) │
                    │  or HNSW (High Precision)   │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │    Top-K Recommendations    │
                    └─────────────────────────────┘
```

## Module Structure

```
hybrid_music_engine/
├── __init__.py         # Package initialization
├── config.py           # Configuration (Paths, Hyperparameters)
├── logger.py           # StepLogger for thesis documentation
├── processors.py       # Data cleaning & BERT tokenization
├── model.py            # Neural Network Architecture
├── trainer.py          # Training loop logic
├── inference.py        # Engine class for Web App
├── requirements.txt    # Dependencies
└── example_usage.py    # Usage examples
```

## Quick Start

### Installation

```bash
cd hybrid_music_engine
pip install -r requirements.txt
```

### Basic Usage

```python
from hybrid_music_engine import MusicRecommendationEngine

# Initialize engine
engine = MusicRecommendationEngine()

# Load trained model and index
engine.load_model("models/best_model.pth")
engine.load_index("models/faiss_index.bin")
engine.load_song_data("data/songs.csv")

# Get recommendations
recommendations = engine.get_similar_songs(
    song_name="Bohemian Rhapsody",
    artist_name="Queen",
    top_k=5
)

for rec in recommendations:
    print(f"{rec['song']} by {rec['artist']} ({rec['similarity']:.2%})")
```

### Training a New Model

```python
from hybrid_music_engine import HybridMusicModel, DataManager, Trainer, get_config

# Initialize
config = get_config()
data_manager = DataManager(config)

# Load and preprocess data
data = data_manager.load_data("data/songs.csv")
data = data_manager.preprocess(data)
train_data, val_data, test_data = data_manager.split_data(data)

# Create and train model
model = HybridMusicModel(config)
trainer = Trainer(model, data_manager, config)
history = trainer.train(train_data, val_data, epochs=10)
```

### Command Line Usage

```bash
# Train model
python example_usage.py train --data data/songs.csv --epochs 10

# Build FAISS index
python example_usage.py index --model models/best_model.pth --data data/songs.csv

# Get recommendations
python example_usage.py recommend --song "Shape of You" --artist "Ed Sheeran"
```

## Features

### Recommendation Modes

1. **Similar Songs** - Content-based similarity using 64D embeddings
2. **Mood-based** - Filter by emotion (happy, sad, energetic, calm)
3. **Context-based** - Filter by activity (party, study, workout, driving)

### Logging for Thesis

All operations are automatically logged to `project_implementation_log.txt`:

```python
from hybrid_music_engine import get_logger

logger = get_logger("project_implementation_log.txt")
logger.log("Loading BERT model", "MODEL")
logger.log_metric("train_loss", 0.234, epoch=5)
```

Sample log output:

```
[2026-01-06 21:30:15.123] [ INFO  ] [  MODEL   ] STARTED: Loading BERT Model
[2026-01-06 21:30:18.456] [SUCCESS] [  MODEL   ] COMPLETED: Loading BERT Model
                                             Details: duration_seconds=3.33
```

## Configuration

```python
from hybrid_music_engine import get_config

config = get_config()

# Modify settings
config.training.epochs = 20
config.training.batch_size = 32
config.training.learning_rate = 2e-5
config.faiss.use_hnsw = False  # Use exact search for accuracy

# BERT settings
config.bert.model_name = "bert-base-uncased"
config.bert.max_length = 256
config.bert.freeze_bert_layers = 10
```

## Model Details

### Input Features

**Audio Features (11 dimensions):**

- energy, danceability, valence, tempo
- acousticness, instrumentalness, speechiness
- liveness, loudness, key, mode

**Text Features:**

- Lyrics (tokenized with BERT)
- Emotion tags

**Metadata:**

- Genre (embedded, 16D)
- Musical Key (embedded, 8D)
- Emotion (embedded, 8D)

### Output

- **64D L2-normalized embedding** for each song
- Used for similarity search via FAISS IndexFlatL2

## Performance Targets

| Metric | Target |
|--------|--------|
| Precision@10 | > 60% |
| NDCG@10 | > 70% |
| Inference Time | < 50ms |
| Songs Indexed | 500K - 1.2M |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- FAISS (cpu or gpu)
- pandas, numpy, scikit-learn

## API Reference

### MusicRecommendationEngine

```python
engine = MusicRecommendationEngine(config)

# Core methods
engine.load_model(model_path)
engine.load_index(index_path)
engine.load_song_data(data_path)

# Recommendations
engine.get_similar_songs(song_name, artist_name, top_k)
engine.get_recommendations_by_mood(mood, top_k)
engine.get_recommendations_by_context(context, top_k)

# Index management
engine.build_index(data_path, save_path)
```

### StepLogger

```python
logger = StepLogger("log_file.txt")

logger.log(message, category, details, level)
logger.log_section("Section Name")
logger.log_start(operation, category)
logger.log_end(operation, category, duration)
logger.log_metric(name, value, epoch)
logger.log_error(message, category, exception)
```

## License

MIT License - See LICENSE file for details.

## Author

Graduation Project - Music Recommendation System
Created: 2026-01-06
