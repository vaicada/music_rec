# Multi-stage Docker build for optimized image size
# Target: < 6GB for Hugging Face Spaces deployment

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal final image
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install libpq5 runtime library (needed by psycopg2-binary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Set PYTHONPATH so all project modules are importable
# /app           → hybrid_music_engine, audio_model, models
# /app/web_app   → database, auth, download_helper
ENV PYTHONPATH=/app:/app/web_app

# Copy all necessary application files
COPY web_app/ ./web_app/
COPY hybrid_music_engine/ ./hybrid_music_engine/
COPY audio_model/ ./audio_model/
COPY models/ ./models/

# Create data directories (populated at runtime by download_helper)
RUN mkdir -p data/processed data/processed/tracks

# Create /data persistent volume directory (SQLite + runtime data on HF Spaces)
RUN mkdir -p /data && chmod 777 /data

# Expose port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Run from /app so all relative imports resolve correctly
WORKDIR /app
CMD ["uvicorn", "web_app.app:app", "--host", "0.0.0.0", "--port", "7860"]
