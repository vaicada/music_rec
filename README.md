---
title: Music Recommender
emoji: 🎵
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
app_port: 7860
---

# 🎵 Music Recommender System

A hybrid music recommendation engine combining content-based filtering (lyrics, audio features) and collaborative filtering concepts.

## Features

- **Search**: Find similar songs by lyrics and audio characteristics
- **Mood**: Get recommendations based on mood (Happy, Sad, Energetic, etc.)
- **Context**: Playlists for specific activities (Workout, Study, Party)
- **YouTube**: Integrated YouTube playback

## Tech Stack

- **Framework**: FastAPI
- **ML**: PyTorch, FAISS, Transformers (BERT)
- **Deployment**: Docker on Hugging Face Spaces

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
cd web_app
uvicorn app:app --reload
```
