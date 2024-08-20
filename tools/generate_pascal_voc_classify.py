import os
import shutil
from xml.etree import ElementTree as ET

def parse_annotation(ann_file):
    tree = ET.parse(ann_file)
    root = tree.getroot()
    objects = root.findall('object')
    objs = []
    for obj in objects:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        # Chuyển đổi số thực thành số nguyên và đảm bảo tọa độ không âm
        xmin = max(0, int(round(xmin)))
        ymin = max(0, int(round(ymin)))
        xmax = max(0, int(round(xmax)))
        ymax = max(0, int(round(ymax)))
        objs.append((name, xmin, ymin, xmax, ymax))
    return objs

def is_valid_image(objs, target_class):
    # Kiểm tra xem ảnh có chỉ chứa đối tượng của lớp mục tiêu hay không
    return all(cls == target_class for cls, _, _, _, _ in objs)

def convert_voc_to_classification(voc_dir, output_dir, classes, train_limit=1500, val_limit=100):
    # Ensure output directories exist
    for subset in ['train', 'val']:
        subset_dir = os.path.join(output_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        for class_name in classes:
            os.makedirs(os.path.join(subset_dir, class_name), exist_ok=True)

    # Path to VOC dataset
    img_dir = os.path.join(voc_dir, 'JPEGImages')
    ann_dir = os.path.join(voc_dir, 'Annotations')

    train_count = {cls: 0 for cls in classes}
    val_count = {cls: 0 for cls in classes}

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    for img_file in img_files:
        ann_file = img_file.replace('.jpg', '.xml')
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, ann_file)
        if not os.path.exists(ann_path):
            continue

        objs = parse_annotation(ann_path)

        for cls in classes:
            if is_valid_image(objs, cls):
                output_subdir = 'train' if train_count[cls] < train_limit else 'val'
                output_class_dir = os.path.join(output_dir, output_subdir, cls)
                shutil.copy(img_path, output_class_dir)
                if train_count[cls] < train_limit:
                    train_count[cls] += 1
                else:
                    val_count[cls] += 1

                if train_count[cls] >= train_limit and val_count[cls] >= val_limit:
                    break

    print("Conversion completed.")

# Usage
voc_dir = '/home/chaos/Documents/ChaosAIVision/dataset/VOCdevkit/VOC2012'
output_dir = '/home/chaos/Documents/ChaosAIVision/dataset'
classes = [ "person"]

convert_voc_to_classification(voc_dir, output_dir, classes)
