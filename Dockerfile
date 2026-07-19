# Production Multi-Stage Dockerfile for ESRGAN Super-Resolution Service
FROM python:3.11-slim

# Install system dependencies for OpenCV and GL libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Generate sample dataset / checkpoints directory
RUN python dataset/download_div2k.py --num_train 5 --num_valid 2

# Expose API (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Command to run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
