import xml.etree.ElementTree as ET
import os
from pathlib import Path
from tqdm import tqdm
import yaml

def convert_label(image_folder_path, voc_label_path, output_folder_path):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    def process_image_and_label(image_id, image_folder_path, voc_label_path, output_folder_path):
        # Define paths
        xml_path = voc_label_path / f'{image_id}.xml'
        output_label_path = output_folder_path / f'{image_id}.txt'
        image_path = image_folder_path / f'{image_id}.jpg'

        # Ensure output folder exists
        output_folder_path.mkdir(parents=True, exist_ok=True)

        # Read and convert label
        with open(xml_path, 'r') as in_file, open(output_label_path, 'w') as out_file:
            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            # Assuming `names` is a dictionary loaded from a YAML file
            with open('/home/chaos/Documents/ChaosAIVision/Build-YOLO-v1-from-scratch/utils/classes.yaml', 'r') as f:
                yaml_data = yaml.safe_load(f)
                names = list(yaml_data['names'].values())

            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls in names and int(obj.find('difficult').text) != 1:
                    xmlbox = obj.find('bndbox')
                    bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
                    cls_id = names.index(cls)  # class id
                    out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + '\n')

        # Move image to output folder
        if image_path.exists():
            output_image_path = output_folder_path / f'{image_id}.jpg'
            image_path.rename(output_image_path)

    # Process all images and labels
    image_ids = [f.stem for f in voc_label_path.glob('*.xml')]

    for image_id in tqdm(image_ids, desc='Processing'):
        process_image_and_label(image_id, image_folder_path, voc_label_path, output_folder_path)

# Example usage
image_folder_path = Path('/home/chaos/Documents/ChaosAIVision/dataset/VOCdevkit/VOC2012/JPEGImages/')  # Replace with your image folder path
voc_label_path = Path('/home/chaos/Documents/ChaosAIVision/dataset/VOCdevkit/VOC2012/Annotations/')  # Replace with your VOC label path
output_folder_path = Path('/home/chaos/Documents/ChaosAIVision/dataset/VOCdevkit/VOC2012/labels/')  # Replace with your output folder path

convert_label(image_folder_path, voc_label_path, output_folder_path)
