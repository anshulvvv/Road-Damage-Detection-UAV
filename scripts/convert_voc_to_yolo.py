import os
import xml.etree.ElementTree as ET
from PIL import Image

classes = ["D00", "D10", "D20", "D40"]

def convert_xml(xml_file, images_dir, labels_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find("filename").text
    image_path = os.path.join(images_dir, filename)
    im = Image.open(image_path)
    width, height = im.size
    lines = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        xmin = float(xmlbox.find("xmin").text)
        ymin = float(xmlbox.find("ymin").text)
        xmax = float(xmlbox.find("xmax").text)
        ymax = float(xmlbox.find("ymax").text)
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        lines.append(f"{cls_id} {x_center} {y_center} {box_width} {box_height}")
    base = os.path.splitext(os.path.basename(xml_file))[0]
    out_file = os.path.join(labels_dir, base + ".txt")
    with open(out_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

def main():
    annotations_dir = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\train\annotations\xmls"
    images_dir = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\train\images"
    labels_dir = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\train\labels"
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    for xml in os.listdir(annotations_dir):
        if xml.endswith(".xml"):
            convert_xml(os.path.join(annotations_dir, xml), images_dir, labels_dir)

if __name__ == "__main__":
    main()
