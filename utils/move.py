import os
import random
import shutil

images_source = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\train\images"
annotations_source = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\train\annotations\xmls"

val_images_folder = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\val"
val_annotations_folder = r"C:\Users\ANSHUL M\Downloads\RDD2022_India\India\val\annotations\xmls"

# Create output folders if they don't exist
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_annotations_folder, exist_ok=True)

# Collect all image filenames (assuming .jpg extension; adjust if needed)
all_images = [f for f in os.listdir(images_source) if f.lower().endswith(".jpg")]
print(f"Found {len(all_images)} images in {images_source}.")

# Randomly pick 50
num_to_select = 50
if len(all_images) < num_to_select:
    print("Not enough images to select 50. Aborting.")
    exit()

random.seed(42)  # for reproducibility
selected_images = random.sample(all_images, num_to_select)

moved_count = 0
for img_name in selected_images:
    # Move the image to val folder
    src_img_path = os.path.join(images_source, img_name)
    dst_img_path = os.path.join(val_images_folder, img_name)
    shutil.move(src_img_path, dst_img_path)

    # Construct corresponding annotation filename 
    # (Assuming YOLO .txt; change '.txt' to '.xml' if that’s your format)
    base_name = os.path.splitext(img_name)[0]
    annotation_name = base_name + ".xml"  # or ".xml"
    
    src_anno_path = os.path.join(annotations_source, annotation_name)
    dst_anno_path = os.path.join(val_annotations_folder, annotation_name)
    # Move annotation if it exists
    if os.path.exists(src_anno_path):
        shutil.move(src_anno_path, dst_anno_path)
    else:
        print(f"[WARNING] Annotation not found for {img_name}: {annotation_name}")
    
    moved_count += 1

print(f"Successfully moved {moved_count} images and their annotations to val folder.")
