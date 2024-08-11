import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from collections import Counter
from utils.metric import non_max_suppression
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from PIL import Image
import io
import os 
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import numpy as np
import pickle


class ManagerDataYaml:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.data = None

    def load_yaml(self) -> dict:
        """
        Load data from YAML file and return its properties as a dictionary.
        """
        try:
            with open(self.yaml_path, 'r') as file:
                self.data = yaml.safe_load(file)
                return self.data
        except Exception as e:
            return f"Error loading YAML file: {self.yaml_path}. Exception: {e}"

    def get_properties(self, key: str) :
        """
        Get the value of a specific property from the loaded YAML data.
        """
        if isinstance(self.data, dict):
            if key in self.data:
                value = self.data[key]
                return (value)
            else:
                return f"Key '{key}' not found in the data."
        else:
            return "Data has not been loaded or is not a dictionary."
        

class ManageSaveDir():
    def __init__(self, data_yaml):
        data_yaml_manage = ManagerDataYaml(data_yaml)
        data_yaml_manage.load_yaml()
        self.save_dir_locations = data_yaml_manage.get_properties('save_dirs')
        self.train_dataset = data_yaml_manage.get_properties('train')
        self.valid_dataset = data_yaml_manage.get_properties('valid')
        self.test_dataset = data_yaml_manage.get_properties('test')
        self.categories = data_yaml_manage.get_properties('categories')

    def create_save_dir(self):
        if not os.path.exists(self.save_dir_locations):
            return f'Folder path {self.save_dir_locations} is not exists'
        else:
            self.result_dir = os.path.join(self.save_dir_locations, 'result')
            weight_dir = os.path.join(self.result_dir, 'weights')
            tensorboard_dir = os.path.join(self.result_dir, 'tensorboard')

            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
                os.makedirs(weight_dir)
                os.makedirs(tensorboard_dir)
                return weight_dir, tensorboard_dir # Cần return tensorbard_dir để lấy location  ghi log và weight
            else:
                counter = 1
                while True:
                    self.result_dir = os.path.join(self.save_dir_locations, f'result{counter}')
                    weight_dir = os.path.join(self.result_dir, 'weights')
                    tensorboard_dir = os.path.join(self.result_dir, 'tensorboard')
                    if not os.path.exists(self.result_dir):
                        os.makedirs(self.result_dir)
                        os.makedirs(weight_dir)
                        os.makedirs(tensorboard_dir)
                        return weight_dir, tensorboard_dir
                    counter += 1
    def get_save_dir_path(self):
        return self.result_dir
    def count_items_in_folder(self, folder_path):
        try:
            items = os.listdir(folder_path)
            num_items = len(items)
            return num_items
        except FileNotFoundError:
            return f'The folder {folder_path} does not exist'
        except PermissionError:
            return f'Permission denied to access the folder {folder_path}'
        
    def count_distribution_labels(self, mode):
        if mode == 'train':
            data_path = self.train_dataset
        elif mode == 'valid':
            data_path = self.valid_dataset
        else:
            data_path = self.test_dataset

        num_categories = []
        for category in self.categories:
            categories_path = os.path.join(data_path, category)
            num_labels = self.count_items_in_folder(categories_path)
            num_categories.append(num_labels)
        return num_categories



    def plot_dataset(self):
        # Lấy số lượng hình ảnh cho các tập dữ liệu
        distribution_train = self.count_distribution_labels('train')
        distribution_valid = self.count_distribution_labels('valid')
        
        num_image_train = sum(distribution_train)
        num_image_valid = sum(distribution_valid)
        try:
            distribution_test = self.count_distribution_labels('test')
            num_image_test = sum(distribution_test)
        except:
            distribution_test = [0] * len(self.categories)  # Sửa tại đây
            num_image_test = 0

        # Tổng hợp số lượng hình ảnh cho mỗi danh mục
        total_distribution = [0] * len(self.categories)
        
        for dist in (distribution_train, distribution_valid, distribution_test):
            for i, count in enumerate(dist):
                if isinstance(count, int):  # Đảm bảo giá trị là số nguyên
                    total_distribution[i] += count
        
        file_path = os.path.join(self.result_dir, 'dataset_distribution.png')

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        # Plot thông tin tổng số hình ảnh
        ax[0].axis('off')
        text_str = f"Number of images in train: {num_image_train}\n" \
                f"Number of images in valid: {num_image_valid}\n" \
                f"Number of images in test: {num_image_test}"
        ax[0].text(0.5, 0.5, text_str, fontsize=12, ha='center', va='center')

        # Plot số lượng hình ảnh tổng hợp cho từng danh mục
        x = range(len(self.categories))
        width = 0.2

        # Tạo màu cho từng danh mục
        cmap = plt.get_cmap('tab20')  # Lấy bảng màu 'tab20'
        colors = [cmap(i) for i in range(len(self.categories))]

        # Vẽ biểu đồ cột với từng màu khác nhau
        ax[1].bar(x, total_distribution, color=colors, width=width, label='Total', align='center')

        ax[1].set_xlabel('Categories')
        ax[1].set_ylabel('Number of images')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(self.categories, rotation=90)
        ax[1].legend()

        plt.tight_layout()

        # Lưu hình ảnh vào tệp tin
        plt.savefig(file_path)
        plt.close()  # Đóng hình ảnh để giải phóng bộ nhớ

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes



def is_image_file(file_path):
    # Danh sách các phần mở rộng ảnh phổ biến
    image_extensions = {'.jpg', '.jpeg', '.png'}
    # Tách phần mở rộng của file
    _, ext = os.path.splitext(file_path.lower())
    return ext in image_extensions

def load_cache(cache_file_path):
    """
    Load cache from a Pickle file.

    Args:
        cache_file_path (str): Path to the cache file.

    Returns:
        list: List of cached data loaded from the file.
    """
    if not os.path.isfile(cache_file_path):
        raise FileNotFoundError(f"The cache file {cache_file_path} does not exist.")
    
    with open(cache_file_path, 'rb') as f:
        cache = pickle.load(f)
    
    return cache