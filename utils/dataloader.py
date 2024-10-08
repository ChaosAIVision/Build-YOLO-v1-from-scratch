import torch
from torch.utils.data import DataLoader
from utils.dataset import YOLODataset
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils.general import ManagerDataYaml
from utils.augmentations import get_train_transforms, get_valid_transforms

class CustomDataLoader(DataLoader):
    def __init__(self,data_yaml, mode :str,  batch_size:int, num_workers:int):


        self.mode = mode
        self.data_yaml = data_yaml
        self.batch_size = batch_size
        self.num_workers = num_workers
     
     
    def create_dataloader(self):

        if self.mode == 'train':
            data = YOLODataset('train', self.data_yaml,S =7, B = 2, C = 3, transform= get_train_transforms())
        elif self.mode == 'valid':
            data = YOLODataset('valid', self.data_yaml, S= 7, B = 2 , C =3, transform= get_valid_transforms())
        else:
            data = YOLODataset('test', self.data_yaml, S= 7, B = 2 , C =3,  transform= get_valid_transforms())
        dataloader = DataLoader(
            dataset= data,
            batch_size= self.batch_size,
            num_workers= self.num_workers,
            pin_memory=True,
            shuffle= True if self.mode == 'train' else False,
            drop_last= False)
        
        return dataloader
     