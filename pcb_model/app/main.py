# app/main.py
import os
from uuid import uuid4
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from pcb_db import save_detection_to_supabase_and_get_urls, get_all_detections

# หา path ของ best.pt แบบไม่ต้องเดา working dir
BASE_DIR = os.path.dirname(__file__)          # โฟลเดอร์ app/
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# จะให้ไฟล์ temp ไปอยู่ใน app/uploads
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="PCB Defect Detection API",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/detect-image")
async def detect_pcb_image(
    file: UploadFile = File(..., description="รูป PCB ที่ต้องการให้บันทึก + ส่ง url + metadata กลับมา"),
):
    """
    - เซฟรูปชั่วคราว
    - รัน YOLO + อัปโหลดรูปหลัก + crop ลง Supabase + insert DB
    - ส่ง JSON กลับมาพร้อม:
        * main_image: ข้อมูลรูป detect หลัก + URL
        * crops: ข้อมูล crop แต่ละอัน + URL + prediction + confidence + bbox
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์รูปภาพเท่านั้น")

    tmp_path = None
    try:
        contents = await file.read()
        filename = f"{uuid4()}_{file.filename}"
        tmp_path = os.path.join(UPLOAD_DIR, filename)

        # เซฟไฟล์ชั่วคราว
        with open(tmp_path, "wb") as f:
            f.write(contents)

        # รัน model + upload Supabase + insert DB + ได้ payload กลับมา
        payload = save_detection_to_supabase_and_get_urls(
            image_path=tmp_path,
            model_path=MODEL_PATH,
            board_code=None,  # ถ้าอยากรับเพิ่มจาก client ค่อยเติม Form field
            note="Created via /detect-image",
        )

        return JSONResponse(payload)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"processing error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@app.get("/detections")
def list_detections():
    """
    ดึงข้อมูล detection ทั้งหมดจาก DB
    - ไม่ส่งข้อมูล crop image
    - มี main image + defects (prediction, confidence, bbox, timestamp)
    """
    try:
        items = get_all_detections()
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db error: {e}")
