FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir tensorflow-cpu==2.17.0

RUN pip install --no-cache-dir \
    fastapi uvicorn pillow numpy \
    python-multipart streamlit requests pymongo pydicom \
    keras opencv-python-headless

COPY app/ ./app/
COPY models/ ./models/
COPY streamlit/ ./streamlit/
COPY samples/ ./samples/

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
