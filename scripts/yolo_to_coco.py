import os
import json
from PIL import Image

# Define your classes; for example, if you only have 4 classes:
categories = [
    {"id": 0, "name": "D00"},
    {"id": 1, "name": "D10"},
    {"id": 2, "name": "D20"},
    {"id": 3, "name": "D40"}
]

def convert_yolo_to_coco(yolo_files, output_json):
    """
    Converts a list of YOLO annotation files to a single COCO JSON file.

    Args:
        yolo_files (list): List of YOLO annotation file paths.
        output_json (str): Path to save the resulting COCO JSON file.
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
        "info": {},
        "licenses": []
    }
    
    annotation_id = 1
    for image_id, label_path in enumerate(yolo_files, start=1):
        # Derive the corresponding image path
        image_path = label_path.replace("labels", "images").replace(".txt", ".jpg")
        if not os.path.exists(image_path):
            print(f"[Warning] Image file not found for label: {label_path}")
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"[Error] Unable to read image {image_path}: {e}")
            continue

        file_name = os.path.basename(image_path)
        coco["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name
        })
        
        # Process the YOLO annotation file
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"[Warning] Skipping malformed line in {label_path}: {line}")
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

    # Save the COCO JSON file
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"[INFO] COCO annotations saved to {output_json}")

def main(yolo_files, output_json):
    """
    Main function to convert YOLO annotations to COCO format.

    Args:
        yolo_files (list): List of YOLO annotation file paths.
        output_json (str): Path to save the resulting COCO JSON file.
    """
    if not yolo_files:
        raise ValueError("[Error] No YOLO annotation files provided.")
    
    if not output_json:
        raise ValueError("[Error] Output JSON path is required.")
    
    convert_yolo_to_coco(yolo_files, output_json)


