# TensorFlow 2.16 supports Python 3.11 and ships Linux wheels for this base.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Runtime libraries used by TensorFlow/Pillow when processing uploaded images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# SQLite user data and newly uploaded images should survive container replacement.
VOLUME ["/app/instance", "/app/static/uploads"]

EXPOSE 5000

# app.py initializes the SQLite database and loads the ML models before serving.
CMD ["python", "app.py"]
