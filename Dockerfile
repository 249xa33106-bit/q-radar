FROM python:3.11-slim

# Install system dependencies for OpenCV and GL graphics
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Generate synthetic preset medical images if missing
RUN python generate_sample_assets.py

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Health check on Streamlit port
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch master production supervisor
CMD ["python", "start_production.py"]
