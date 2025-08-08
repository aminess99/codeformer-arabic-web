FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_railway.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install torch first (required for basicsr)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements_railway.txt

# Copy the rest of the application
COPY . .

# Install local basicsr package
RUN pip install -e ./basicsr

# Create necessary directories
RUN mkdir -p weights/CodeFormer weights/facelib weights/realesrgan weights/gfpgan

# Expose port
EXPOSE 5000

# Start the application
CMD ["python", "simple_web_server.py"]