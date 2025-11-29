# pcb_model.py
import os
from io import BytesIO

from ultralytics import YOLO
from PIL import Image


def run_pcb_detection(
    image_path: str,
    model_path: str = "best.pt",
):
    """
    รัน YOLO กับรูป PCB 1 รูป
    * ไม่ยุ่งกับ Supabase
    * คืนผลลัพธ์เป็น dict ที่มี
      - annotated_image: bytes + meta
      - crops: list ของ defect crop (bytes + prediction + confidence + bbox)
    """
    # 1) โหลดโมเดล YOLO
    model = YOLO(model_path)

    # 2) โหลดรูป input (raw image) ด้วย PIL
    img = Image.open(image_path).convert("RGB")

    # 3) รัน YOLO
    results = model(img)[0]

    # 4) ทำ annotated image (วาดกล่อง defect ลงรูปหลัก)
    annotated_arr = results.plot()
    annotated_img = Image.fromarray(annotated_arr)

    main_buf = BytesIO()
    annotated_img.save(main_buf, format="PNG")
    main_buf.seek(0)
    main_bytes = main_buf.getvalue()

    # สร้างโครงสำหรับ output
    detection_result = {
        "annotated_image": {
            "bytes": main_bytes,
            "width": annotated_img.width,
            "height": annotated_img.height,
            "original_filename": os.path.basename(image_path),
        },
        "crops": [],
    }

    print("\n==== DETECTIONS (from model) ====")
    if not results.boxes:
        print("No defects detected.")
        return detection_result

    # 5) loop กล่องทีละอัน → crop + เก็บ prediction/confidence + bbox
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls.item())
        cls_name = results.names[cls_id]
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        print(f"- {cls_name} ({conf:.2f}) box: {x1},{y1},{x2},{y2}")

        crop_img = img.crop((x1, y1, x2, y2))

        crop_buf = BytesIO()
        crop_img.save(crop_buf, format="PNG")
        crop_buf.seek(0)
        crop_bytes = crop_buf.getvalue()

        crop_info = {
            "bytes": crop_bytes,
            "width": crop_img.width,
            "height": crop_img.height,
            "prediction": cls_name,
            "confidence": conf,
            "bbox": {
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
            },
        }
        detection_result["crops"].append(crop_info)

    return detection_result


if __name__ == "__main__":
    # ตัวอย่างใช้รันเดี่ยว ๆ เพื่อเช็คว่า model ทำงาน
    result = run_pcb_detection("test1.png", model_path="best.pt")
    print("\nSummary:")
    print(" Annotated image size:", result["annotated_image"]["width"], "x", result["annotated_image"]["height"])
    print(" Num crops:", len(result["crops"]))
