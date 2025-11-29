FROM python:3.11-slim

# ตั้งค่าเบื้องต้น
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ติดตั้ง system deps ที่จำเป็นสำหรับ ultralytics + opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ติดตั้ง Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดแอปเข้า container
COPY app/ .

# เปิด port 8000 ใน container
EXPOSE 8000

# รัน FastAPI app (main.py อยู่ในโฟลเดอร์ app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
