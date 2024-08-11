import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from utils.general import ManagerDataYaml, is_image_file
from utils.augmentations import transform_labels_to_one_hot

import os
import pickle
from tqdm import tqdm  # Thư viện để tạo thanh tiến trình

import torch
from torch.utils.data import Dataset
from utils.general import ManagerDataYaml, is_image_file, load_cache




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
    def __init__(self, is_train, data_yaml, S= 7, B= 2, C = 20, transform= None):
        data_yaml_manage = ManagerDataYaml(data_yaml)
        data_yaml_manage.load_yaml()

        self.transform = transform
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
        image = Image.open(infor['images'])
        labels = infor['labels']
        boxes = torch.tensor(labels)

         # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix.shape




        




     



               







            


        


