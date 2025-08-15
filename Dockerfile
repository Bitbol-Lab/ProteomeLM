# =====================================================================
#                        ProteomeLM Docker Container
# =====================================================================
# Simple Docker container for ProteomeLM training and inference
# =====================================================================

FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set basic environment
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8

# Set timezone
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install ProteomeLM
RUN pip install -e .

# Create directories
RUN mkdir -p data output weights logs

# Default command
CMD ["bash"]