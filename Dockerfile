# Use a lightweight Python base image
FROM python:3.12-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Install `uv` for dependency management
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure `uv` is on PATH
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py .  
COPY tasksA.py tasksB.py .  
COPY requirements.txt .  

# Install dependencies using `uv`
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI application using `uvicorn`
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]