FROM python:3.11-slim

WORKDIR /app

# install system deps that some Python packages need to build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy only what the app needs
COPY main.py .
COPY api/ api/
COPY src/ src/
COPY model/ model/
COPY frontend/ frontend/
COPY data/clap_db/ data/clap_db/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]