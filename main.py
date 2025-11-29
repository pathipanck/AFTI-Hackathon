# api.py
import os
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from pcb_model.pcb_db import save_detection_to_supabase  # ใช้ฟังก์ชันที่มีอยู่แล้ว

app = FastAPI(
    title="PCB Defect Detection API",
    version="1.0.0",
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/detect")
async def detect_pcb(
    file: UploadFile = File(..., description="รูป PCB ที่ต้องการตรวจ"),
    board_code: str | None = Form(None),
    note: str | None = Form(None),
):
    """
    รับรูป PCB จาก client → รัน YOLO + เก็บผลลง Supabase DB + Storage
    """
    # 1) เช็คว่าเป็นไฟล์รูปไหม
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์รูปภาพเท่านั้น")

    # 2) เซฟไฟล์ชั่วคราวลงดิสก์
    try:
        contents = await file.read()
        filename = f"{uuid4()}_{file.filename}"
        tmp_path = os.path.join(UPLOAD_DIR, filename)

        with open(tmp_path, "wb") as f:
            f.write(contents)

        # 3) เรียกฟังก์ชันที่เรามีอยู่แล้วให้จัดการทั้งหมด
        #    - รัน YOLO (pcb_model.run_pcb_detection)
        #    - อัพโหลดรูป detect หลัก + crop ไป Supabase Storage
        #    - insert ลง pcb_main_images + pcb_defect_crops
        # ตอนนี้ save_detection_to_supabase ยังไม่ return อะไร เราใช้เป็น side-effect ไปก่อน
        save_detection_to_supabase(
            image_path=tmp_path,
            model_path="pcb_model/best.pt",
            board_code=board_code,
            note=note,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"processing error: {e}")
    finally:
        # 4) ลบไฟล์ชั่วคราวออก (ถ้าอยากเก็บไว้ debug ก็ comment บรรทัดนี้ได้)
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # 5) ส่ง response กลับไปแบบง่าย ๆ ก่อน
    return JSONResponse(
        {
            "status": "ok",
            "message": "Image processed and saved to Supabase.",
            "board_code": board_code,
            "note": note,
            "original_filename": file.filename,
        }
    )
