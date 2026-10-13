# Pump display OCR server for Linux.
# NOT YET BUILT OR TESTED: written on a machine without Docker.
FROM python:3.11-slim

# Shared libraries that OpenCV and PaddlePaddle need on a slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config
# The recognition model fine-tuned on pump digits (not in git: copy the
# models folder next to this file before building).
COPY models ./models
# The image has the CPU build of PaddlePaddle.
ENV PUMP_OCR_DEVICE=cpu

# Download the OCR models into the image now, so the running server never
# needs internet access. Only models are downloaded; no image is sent anywhere.
RUN PADDLE_PDX_MODEL_SOURCE=huggingface python -c \
    "import numpy as np; from app.ocr.engine import detect; detect(np.full((64, 64, 3), 255, np.uint8))"

# Scans (pictures + database) live here. Mount a volume so they survive
# container restarts and upgrades.
ENV PUMP_OCR_DATA_DIR=/data
VOLUME /data

EXPOSE 8000
# PUMP_OCR_ACCESS_CODE must be set at run time; the server refuses to start
# without it. HTTPS is provided by the reverse proxy in docker-compose.yml.
CMD ["uvicorn", "app.main:production_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
