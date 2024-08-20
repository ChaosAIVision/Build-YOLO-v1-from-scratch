import torch
import os
import torchvision
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from utils.general import ManagerDataYaml, is_image_file
from utils.augmentations import YOLOAugmentation

import os
import pickle
from tqdm import tqdm  # Thư viện để tạo thanh tiến trình

import torch
from torch.utils.data import Dataset
from utils.general import ManagerDataYaml, is_image_file, load_cache
from torchvision import transforms
import numpy as np




class Create_YOLO_Cache:
    def __init__(self, is_train, data_yaml):
        # Load paths from YAML
        data_yaml_manage = ManagerDataYaml(data_yaml)
        data_yaml_manage.load_yaml()
        
        if is_train == 'train':
            images_path = data_yaml_manage.get_properties(key='train')
        elif is_train == 'valid':
            images_path = data_yaml_manage.get_properties(key='valid')
        else:
            raise ValueError("Invalid value for 'is_train'. Must be 'train' or 'valid'.")
        
        labels_path = images_path.replace('images', 'labels')

        self.image_path_list = []
        self.labels_path_list = []

        for file_name in os.listdir(images_path):
            image_path = os.path.join(images_path, file_name)
            if is_image_file(image_path):
                label_path = os.path.join(labels_path, file_name)
                path_tmp, ext = os.path.splitext(label_path)
                label_path = path_tmp + '.txt'
                if os.path.isfile(label_path):
                    self.image_path_list.append(image_path)
                    self.labels_path_list.append(label_path)
        
               
    def __save_cache__(self):
    

        cache = []

        for index in tqdm(range(len(self.labels_path_list)), desc='Loading Cache', unit='file'):
            label_path = self.labels_path_list[index]
            image_path = self.image_path_list[index]            
            boxes = []
            try:
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:  # Ensure there are 5 parts (class_id, x_center, y_center, width, height)
                            class_id, x_center, y_center, width, height = parts
                            class_id = int(class_id)
                            x_center = float(x_center)
                            y_center = float(y_center)
                            width = float(width)
                            height = float(height)
                            boxes.append([class_id, x_center, y_center, width, height])
            except Exception as e:
                print(f"Error reading label file {label_path}: {e}")
                continue
            
            target = {'images': image_path, 'labels': boxes}
            cache.append(target)
        
        # Define the path to save the cache, lùi lại một cấp
        parent_dir = os.path.dirname(os.path.dirname(self.labels_path_list[0]))
        cache_file = os.path.join(parent_dir, 'labels.cache')
        
        # Save cache to a Pickle file
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        
        print(f"Cache saved to {cache_file}")



        
class YOLODataset(Dataset):
    def __init__(self, is_train, data_yaml, S= 7, B= 2, C = 3, transform= None):
        data_yaml_manage = ManagerDataYaml(data_yaml)
        data_yaml_manage.load_yaml()
        self.is_train = is_train
        self.transform = transform
        self.to_tensor =  torchvision.transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

        self.S = S 
        self.B = B 
        self.C = C
     
        if is_train == 'train':
            images_path = data_yaml_manage.get_properties(key='train')
        elif is_train == 'valid':
            images_path = data_yaml_manage.get_properties(key='valid')
        else:
            images_path = data_yaml_manage.get_properties(key='test')

        cache_path =  images_path.replace('images', 'labels.cache')
        self.cache = load_cache(cache_path)
      
       

    def __len__(self):
        return len(self.cache)
    
    
    def __getitem__(self, index):
        infor = self.cache[index]
        image = Image.open(infor['images']).convert("RGB")
        labels = infor['labels']
        labels = np.array(labels)
        just_boxes = labels[:,1:]
        labels = labels[:,0]
        try:
            if  self.transform:
                        sample = {
                        'image':  np.array(image),
                        'bboxes': just_boxes,
                        'labels': labels
                        }

                        sample =  self.transform(**sample)
                        image = sample['image']
                        boxes = sample['bboxes']
                        labels = sample['labels']
        except:
            boxes = just_boxes
            labels = labels
            image = self.to_tensor(image)



        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        image  = torch.as_tensor(image, dtype=torch.float32)

        # Iterate through each bounding box in YOLO format.
        for box, class_label in zip(boxes, labels):
            x, y, width, height = box.tolist()
            class_label = int(class_label)

            # Calculate the grid cell (i, j) that this box belongs to.
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            # Calculate the width and height of the box relative to the grid cell.
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object has been found in this specific cell (i, j) before:
            if label_matrix[i, j, 3] == 0:
                # Mark that an object exists in this cell.
                label_matrix[i, j, 3] = 1

                # Store the box coordinates as an offset from the cell boundaries.
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                # Set the box coordinates in the label matrix.
                label_matrix[i, j, 4:8] = box_coordinates

                # Set the one-hot encoding for the class label.
                label_matrix[i, j, class_label] = 1
                

        return image, label_matrix



        




     



               







            


        


