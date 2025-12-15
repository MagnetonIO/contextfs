# ContextFS Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml setup.sh ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[web,postgres]"

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 8765

# Set environment variables
ENV CONTEXTFS_DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

# Run the web server
CMD ["python", "-m", "contextfs.web.server", "--host", "0.0.0.0", "--port", "8765"]
