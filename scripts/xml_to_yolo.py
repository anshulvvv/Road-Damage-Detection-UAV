import os
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(input_dir, output_dir, class_name_to_id):
    """
    Converts XML annotations to YOLO format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".xml"):
            continue

        xml_path = os.path.join(input_dir, file_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image dimensions
        size = root.find("size")
        if size is None:
            print(f"[Warning] No size info in {xml_path}, skipping")
            continue

        width = float(size.find("width").text)
        height = float(size.find("height").text)

        # Prepare output path
        txt_file = file_name.replace(".xml", ".txt")
        txt_path = os.path.join(output_dir, txt_file)
        yolo_lines = []

        # Process each object
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()
            
            if class_name not in class_name_to_id:
                print(f"[Warning] Unknown class {class_name} in {xml_path}")
                continue

            class_id = class_name_to_id[class_name]
            bndbox = obj.find("bndbox")

            try:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
            except AttributeError:
                print(f"[Error] Invalid bndbox in {xml_path}")
                continue

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )

        # Write YOLO file
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))
            
        print(f"Converted {xml_path} -> {txt_path}")

def main(input_dir, output_dir):
    # Default class mapping (customize as needed)
    class_name_to_id = {
        "D00": 0,
        "D10": 1,
        "D20": 2,
        "D40": 3
    }

    # Validate paths
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist!")

    convert_xml_to_yolo(
        input_dir=input_dir,
        output_dir=output_dir,
        class_name_to_id=class_name_to_id
    )

