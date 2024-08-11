from torchvision.transforms import ToTensor, Compose, Normalize, RandomAffine, ColorJitter, Resize
from torch import tensor
import torch

def transform_input(image_size:int, is_train: bool) -> Compose:
    if is_train:

        transform = Compose([
                ToTensor(),
                Resize((image_size,image_size)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                RandomAffine(
                    degrees=(-5, 5),
                    translate=(0.15, 0.15),
                    scale=(0.85, 1.15),
                    shear=10
                ),
                ColorJitter(
                    brightness=0.125,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.05
                )
        
                ])
    else:
        transform = Compose([
                ToTensor(),
                Resize((image_size,image_size)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    return transform


def transform_labels_to_one_hot(labels: list, num_classes:int) -> torch.LongTensor:
    labels_tensor = torch.tensor(labels)
    one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes= num_classes)
    return one_hot.to(torch.long)


