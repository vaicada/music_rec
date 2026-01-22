# Multi-stage Docker build for optimized image size
# Target: < 6GB for Railway deployment

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
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

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy only necessary application files
COPY web_app/ ./web_app/
COPY hybrid_music_engine/ ./hybrid_music_engine/
COPY models/ ./models/

# Create data directory (will be populated at runtime)
RUN mkdir -p data/processed

# Expose port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Start command
CMD cd web_app && uvicorn app:app --host 0.0.0.0 --port 7860
