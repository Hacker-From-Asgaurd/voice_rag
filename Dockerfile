# Dockerfile for VOICE-RAG-HHGOA Deployment
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data assets
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/

# Environment settings
ENV PYTHONPATH="/app/src:/app"
ENV PORT=8000
ENV HOST=0.0.0.0

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
