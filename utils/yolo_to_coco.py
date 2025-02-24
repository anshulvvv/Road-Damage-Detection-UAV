import os
import json
from glob import glob
from PIL import Image

# Set these paths as appropriate
images_dir = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\train\images"
labels_dir = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\train\annotations\labels"  # YOLO-format txt files
output_json = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\train\annotations\annotations_train_coco.json"

# Define your classes; for example, if you only have 4 classes:
categories = [
    {"id": 0, "name": "D00"},
    {"id": 1, "name": "D10"},
    {"id": 2, "name": "D20"},
    {"id": 3, "name": "D40"}
]

def convert_yolo_to_coco():
    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
        "info": {},
        "licenses": []
    }
    
    annotation_id = 1
    image_files = sorted(glob(os.path.join(images_dir, "*.*")))
    for image_id, image_path in enumerate(image_files, start=1):
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            continue

        file_name = os.path.basename(image_path)
        coco["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name
        })
        
        # Assume corresponding YOLO label file has same base name with .txt extension
        label_path = os.path.join(labels_dir, os.path.splitext(file_name)[0] + ".txt")
        if not os.path.exists(label_path):
            continue
        
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w_norm, h_norm = map(float, parts)
                x_center *= width
                y_center *= height
                box_width = w_norm * width
                box_height = h_norm * height
                x_min = x_center - box_width / 2
                y_min = y_center - box_height / 2
                area = box_width * box_height
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"COCO annotations saved to {output_json}")

if __name__ == "__main__":
    convert_yolo_to_coco()
