# pcb_db.py
import os
import uuid

from dotenv import load_dotenv
from supabase import create_client, Client

from pcb_model import run_pcb_detection  # import จากไฟล์แรก

from typing import List, Dict, Any

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # หรือ SUPABASE_KEY ถ้าใช้ชื่ออื่น
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "pcb-images")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------- Helper: upload to Storage ----------

def upload_to_storage(bytes_data: bytes, folder: str, ext: str = "png") -> tuple[str, str]:
    """
    อัพโหลดไฟล์ไป Supabase Storage
    return (storage_path, public_url)
    """
    filename = f"{uuid.uuid4().hex}.{ext}"
    storage_path = f"{folder}/{filename}"

    # ถ้า error มันจะ throw exception เอง
    supabase.storage.from_(BUCKET_NAME).upload(
        path=storage_path,
        file=bytes_data,
        file_options={"content-type": f"image/{ext}"},
    )

    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)
    return storage_path, public_url


# ---------- DB Insert Helpers ----------

def insert_main_image(
    storage_path: str,
    public_url: str,
    width: int,
    height: int,
    original_filename: str | None = None,
    board_code: str | None = None,
    note: str | None = None,
) -> str:
    """
    Insert row ลง pcb_main_images แล้วคืน id (string)
    """
    data = {
        "storage_path": storage_path,
        "public_url": public_url,
        "width": width,
        "height": height,
        "original_filename": original_filename,
        "board_code": board_code,
        "note": note,
    }
    res = supabase.table("pcb_main_images").insert(data).execute()
    row = res.data[0]
    return row["id"]


def insert_defect_crop(
    main_image_id: str,
    crop_storage_path: str,
    crop_public_url: str,
    crop_width: int,
    crop_height: int,
    prediction: str,
    confidence: float,
    bbox: dict | None = None,
):
    """
    Insert row ลง pcb_defect_crops
    bbox: dict เช่น {"x": 100, "y": 120, "w": 50, "h": 40} หรือ None
    """
    data = {
        "main_image_id": main_image_id,
        "crop_storage_path": crop_storage_path,
        "crop_public_url": crop_public_url,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "prediction": prediction,
        "confidence": confidence,
    }

    if bbox:
        data["bbox_x"] = bbox.get("x")
        data["bbox_y"] = bbox.get("y")
        data["bbox_width"] = bbox.get("w")
        data["bbox_height"] = bbox.get("h")

    res = supabase.table("pcb_defect_crops").insert(data).execute()
    return res.data[0]

def save_detection_to_supabase_and_get_urls(
    image_path: str,
    model_path: str,
    board_code: str | None = None,
    note: str | None = None,
):
    """
    รัน YOLO, upload รูปหลัก + crop ไป Supabase, insert DB
    แล้วคืน payload ที่มี URL + metadata กลับมา
    """
    detection_result = run_pcb_detection(
        image_path=image_path,
        model_path=model_path,
    )

    annotated = detection_result["annotated_image"]

    # 1) upload main image
    main_storage_path, main_public_url = upload_to_storage(
        annotated["bytes"],
        folder="pcb/main",
        ext="png",
    )

    # 2) insert main image row
    main_image_id = insert_main_image(
        storage_path=main_storage_path,
        public_url=main_public_url,
        width=annotated["width"],
        height=annotated["height"],
        original_filename=annotated["original_filename"],
        board_code=board_code,
        note=note,
    )

    main_payload = {
        "id": main_image_id,
        "storage_path": main_storage_path,
        "public_url": main_public_url,
        "width": annotated["width"],
        "height": annotated["height"],
        "original_filename": annotated["original_filename"],
        "board_code": board_code,
        "note": note,
    }

    # 3) upload crops + insert defects + สร้าง payload
    crops_payload = []
    for crop in detection_result["crops"]:
        crop_storage_path, crop_public_url = upload_to_storage(
            crop["bytes"],
            folder="pcb/crops",
            ext="png",
        )

        defect_row = insert_defect_crop(
            main_image_id=main_image_id,
            crop_storage_path=crop_storage_path,
            crop_public_url=crop_public_url,
            crop_width=crop["width"],
            crop_height=crop["height"],
            prediction=crop["prediction"],
            confidence=crop["confidence"],
            bbox=crop["bbox"],
        )

        crops_payload.append(
            {
                "id": defect_row["id"],
                "crop_storage_path": crop_storage_path,
                "crop_public_url": crop_public_url,
                "width": crop["width"],
                "height": crop["height"],
                "prediction": crop["prediction"],
                "confidence": crop["confidence"],
                "bbox": crop["bbox"],
            }
        )

    return {
        "main_image": main_payload,
        "crops": crops_payload,
    }

def get_all_detections() -> List[Dict[str, Any]]:
    # ดึงรูปหลักทั้งหมด
    main_res = supabase.table("pcb_main_images").select("*").execute()
    main_rows = main_res.data or []

    # ดึง defects ทั้งหมด
    crop_res = supabase.table("pcb_defect_crops").select("*").execute()
    crop_rows = crop_res.data or []

    # group crop ตาม main_image_id
    crops_by_main: Dict[str, List[Dict[str, Any]]] = {}
    for c in crop_rows:
        mid = c.get("main_image_id")
        if mid is None:
            continue
        crops_by_main.setdefault(mid, []).append(c)

    results: List[Dict[str, Any]] = []

    for m in main_rows:
        mid = m["id"]
        main_item: Dict[str, Any] = {
            "main_image_id": mid,
            "main_image_url": m.get("public_url"),
            "storage_path": m.get("storage_path"),
            "original_filename": m.get("original_filename"),
            "board_code": m.get("board_code"),
            "note": m.get("note"),
            # ถ้า table มีคอลัมน์ created_at (ตาม pattern ของ Supabase) ก็จะติดมาด้วย
            "timestamp": m.get("created_at"),
            "defects": [],
        }

        defects = []
        for c in crops_by_main.get(mid, []):
            defects.append(
                {
                    "id": c.get("id"),
                    "prediction": c.get("prediction"),
                    "confidence": c.get("confidence"),
                    "bbox": {
                        "x": c.get("bbox_x"),
                        "y": c.get("bbox_y"),
                        "w": c.get("bbox_width"),
                        "h": c.get("bbox_height"),
                    },
                    "timestamp": c.get("created_at"),
                }
            )

        main_item["defects"] = defects
        results.append(main_item)

    return results
