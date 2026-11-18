# Production Dockerfile for TruthLens AI Backend
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU-only PyTorch (reduces build image by ~2 GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, models, and helper modules
COPY backend/ ./backend/
COPY models/ ./models/
COPY src/ ./src/

# Default environment configuration
ENV HOST=0.0.0.0
ENV PORT=7860
ENV CORS_ORIGINS=*
ENV LLM_PROVIDER=offline

# Hugging Face Spaces uses 7860, Cloud platforms like Render/Railway pass $PORT
EXPOSE 7860 8000 10000

# Shell form ensures $PORT is dynamically evaluated if set by the hosting platform
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}
