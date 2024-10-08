from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class YOLOAugmentation:
    def __init__(self,is_train,image_size=(448, 448)):
        self.image_size = image_size
        self.geometric_transform = self.get_geometric_transform()
        self.color_transform = self.get_color_transform()
        self.get_valid_transform = self.get_valid_transform()
        self.is_train = is_train

    def get_geometric_transform(self):
        return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0),
            A.HorizontalFlip(p=0.1),
            A.VerticalFlip(p=0.1),
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def get_valid_transform(self):
        return A.Compose([A.Resize(448, 448)],  bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def get_color_transform(self):
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.1),
        ])

    def pil_to_numpy(self, image):
        return np.array(image)

    def numpy_to_pil(self, image):
        return Image.fromarray(image)

    def transform_image(self, image, bboxes):
        # Convert PIL image to numpy array
        image_np = self.pil_to_numpy(image)
        
        # Convert bboxes
        class_labels = [bbox[0] for bbox in bboxes]
        bboxes = [bbox[1:] for bbox in bboxes]
        
        # Apply geometric transformation
        if self.is_train == 'train':
            transformed = self.geometric_transform(image=image_np, bboxes=bboxes, class_labels=class_labels)
        else:
            transformed = self.get_valid_transform(image=image_np, bboxes=bboxes, class_labels=class_labels)
        image_np = transformed['image']
        bboxes = transformed['bboxes']
        class_labels = transformed['class_labels']
        
        # Combine class labels back with bboxes
        bboxes = [[label] + list(bbox) for label, bbox in zip(class_labels, bboxes)]
        
        # Convert numpy array back to PIL image
        image_pil = self.numpy_to_pil(image_np)
        
        return image_pil, bboxes

    def transform_color(self, image):
        # Convert PIL image to numpy array
        image_np = self.pil_to_numpy(image)
        
        # Apply color transformation
        image_np = self.color_transform(image=image_np)['image']
        
        # Convert numpy array back to PIL image
        image_pil = self.numpy_to_pil(image_np)
        
        return image_pil

    def __call__(self, image, bboxes):
        image, bboxes = self.transform_image(image, bboxes)
        if self.is_train == 'train':
            image = self.transform_color(image)
        return image, bboxes



def get_train_transforms(WIDTH = 448, HEIGHT = 448 ):
    return A.Compose([A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),
                      A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],p=0.9),
                      A.ToGray(p=0.01),
                      A.HorizontalFlip(p=0.2),
                      A.VerticalFlip(p=0.2),
                      A.Resize(height=WIDTH, width=HEIGHT, p=1),
                    #   A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
                      ToTensorV2(p=1.0)],
                      p=1.0,
                      bbox_params=A.BboxParams(format='yolo', min_area=0, min_visibility=0, label_fields=['labels'])
                      )

def get_valid_transforms(WIDTH = 448, HEIGHT = 448):
    return A.Compose([A.Resize(height=WIDTH, width=HEIGHT, p=1.0),
                      ToTensorV2(p=1.0)],
                      p=1.0,
                      bbox_params=A.BboxParams(format='yolo', min_area=0, min_visibility=0, label_fields=['labels'])
                      )
