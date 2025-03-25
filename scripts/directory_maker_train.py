import os
import random
import shutil
import json
import argparse
from xml_to_yolo import main
from yolo_to_coco import main

def xml_to_yolo(xml_path, yolo_path):
    """Convert XML annotation to YOLO format """
    xml_to_yolo.main(xml_path, yolo_path)

def yolo_to_coco(yolo_files, output_path):
    """Convert YOLO annotations to COCO format (placeholder implementation)"""
    yolo_to_coco.main(yolo_files, output_path)

def main(base_dir, val_split=0.1):
    # Original training data paths
    orig_train_images = os.path.join(base_dir, 'train', 'images')
    orig_train_annotations = os.path.join(base_dir, 'train', 'annotations')

    # Create YOLO directory structure
    base_dir1 = os.path.join(base_dir, 'yolo')
    yolo_labels_train = os.path.join(base_dir1, 'labels', 'train')
    yolo_labels_val = os.path.join(base_dir1, 'labels', 'val')
    yolo_images_train = os.path.join(base_dir1, 'images', 'train')
    yolo_images_val = os.path.join(base_dir1, 'images', 'val')
    
    os.makedirs(yolo_labels_train, exist_ok=True)
    os.makedirs(yolo_labels_val, exist_ok=True)
    os.makedirs(yolo_images_train, exist_ok=True)
    os.makedirs(yolo_images_val, exist_ok=True)

    base_dir2 = os.path.join(base_dir, 'rtmdetr')
    # Create RTM-DET/DETR structure
    rtm_train_images = os.path.join(base_dir2, 'train', 'images')
    rtm_val_images = os.path.join(base_dir2, 'val', 'images')
    os.makedirs(rtm_train_images, exist_ok=True)
    os.makedirs(rtm_val_images, exist_ok=True)

    # List and split images
    all_images = [f for f in os.listdir(orig_train_images) 
                 if os.path.isfile(os.path.join(orig_train_images, f))]
    
    val_count = int(len(all_images) * val_split)
    val_images = set(random.sample(all_images, val_count))

    # Create symlinks/copies
    for image in all_images:
        src = os.path.abspath(os.path.join(orig_train_images, image))
        is_val = image in val_images
        
        # YOLO paths
        yolo_dest = os.path.join(yolo_images_val if is_val else yolo_images_train, image)
        # RTM paths
        rtm_dest = os.path.join(rtm_val_images if is_val else rtm_train_images, image)
        
        # Create links for both frameworks
        for dest in [yolo_dest, rtm_dest]:
            try:
                os.symlink(src, dest)
            except Exception as e:
                print(f"Couldn't create symlink {src} → {dest}: {str(e)}")
                # Uncomment to enable copy fallback
                # shutil.copy2(src, dest)

    # Process annotations
    xml_files = [f for f in os.listdir(orig_train_annotations) if f.endswith('.xml')]
    for xml_file in xml_files:
        base_name = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(orig_train_annotations, xml_file)
        
        # Find matching image
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        val_match = any(base_name + ext in val_images for ext in image_extensions)
        
        yolo_path = os.path.join(yolo_labels_val if val_match else yolo_labels_train, 
                               base_name + '.txt')
        xml_to_yolo(xml_path, yolo_path)

    # Convert to COCO format
    yolo_to_coco(
        [os.path.join(yolo_labels_train, f) for f in os.listdir(yolo_labels_train)],
        os.path.join(base_dir, 'train', 'annotations','annotations_coco.json')
    )
    yolo_to_coco(
        [os.path.join(yolo_labels_val, f) for f in os.listdir(yolo_labels_val)],
        os.path.join(base_dir, 'val','annotations', 'annotations_coco.json')
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Organize dataset for object detection models')
    parser.add_argument('--base_dir', type=str, required=True, 
                      help='Root directory of the dataset')
    parser.add_argument('--val_split', type=float, default=0.1,
                      help='Validation split ratio (default: 0.1)')
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    # Validate paths
    if not os.path.exists(args.base_dir):
        raise ValueError(f"Base directory {args.base_dir} does not exist!")
    
    required = [
        ('train', 'images'),
        ('train', 'annotations')
    ]
    for path in required:
        full_path = os.path.join(args.base_dir, *path)
        if not os.path.exists(full_path):
            raise ValueError(f"Missing required directory: {full_path}")

    main(base_dir=args.base_dir, val_split=args.val_split)
