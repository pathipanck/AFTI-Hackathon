# pcb_db.py
import os
import uuid

from dotenv import load_dotenv
from supabase import create_client, Client

from pcb_model import run_pcb_detection  # import จากไฟล์แรก

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


# ---------- Orchestrator: ใช้ model + เก็บ DB ----------

def save_detection_to_supabase(
    image_path: str,
    model_path: str = "best.pt",
    board_code: str | None = None,
    note: str | None = None,
):
    """
    1) เรียก model จาก pcb_model.run_pcb_detection
    2) อัพโหลดรูป detect หลัก
    3) อัพโหลด crop ทุกรูป
    4) Insert DB ทั้ง pcb_main_images และ pcb_defect_crops
    """

    # 1) รัน model (ไม่ยุ่งกับ Supabase)
    detection_result = run_pcb_detection(image_path=image_path, model_path=model_path)

    annotated = detection_result["annotated_image"]
    main_bytes = annotated["bytes"]
    main_width = annotated["width"]
    main_height = annotated["height"]
    original_filename = annotated["original_filename"]

    # 2) upload main image
    main_storage_path, main_public_url = upload_to_storage(
        main_bytes,
        folder="pcb/main",
        ext="png",
    )

    # 3) insert main image row
    main_image_id = insert_main_image(
        storage_path=main_storage_path,
        public_url=main_public_url,
        width=main_width,
        height=main_height,
        original_filename=original_filename,
        board_code=board_code,
        note=note,
    )

    print("✅ Main image uploaded & inserted:")
    print("   id:", main_image_id)
    print("   url:", main_public_url)

    # 4) loop crops → upload + insert defect row
    crops = detection_result["crops"]
    print(f"\n==== Saving {len(crops)} defect crops to Supabase ====")

    for idx, crop in enumerate(crops, start=1):
        crop_bytes = crop["bytes"]
        crop_width = crop["width"]
        crop_height = crop["height"]
        prediction = crop["prediction"]
        confidence = crop["confidence"]
        bbox = crop["bbox"]

        crop_storage_path, crop_public_url = upload_to_storage(
            crop_bytes,
            folder="pcb/crops",
            ext="png",
        )

        defect_row = insert_defect_crop(
            main_image_id=main_image_id,
            crop_storage_path=crop_storage_path,
            crop_public_url=crop_public_url,
            crop_width=crop_width,
            crop_height=crop_height,
            prediction=prediction,
            confidence=confidence,
            bbox=bbox,
        )

        print(
            f"  → Defect #{idx} row id: {defect_row['id']}, "
            f"class={prediction}, conf={confidence:.4f}"
        )
        print("    crop url:", crop_public_url)

    print("\n🎉 Done: YOLO results have been uploaded & saved to DB.")


if __name__ == "__main__":
    # ใช้งานจริงจากไฟล์นี้ตัวเดียวก็ได้
    save_detection_to_supabase(
        image_path="test1.png",     # เปลี่ยนให้ตรงกับไฟล์จริง
        model_path="best.pt",       # โมเดลของเพื่อน
        board_code="BOARD-0001",
        note="Test upload from split files",
    )
