from models.yolo import Yolov1
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from utils.general import ManagerDataYaml, ManageSaveDir, cellboxes_to_boxes, convert_xywh2xyxy
from utils.metric import  non_max_suppression
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



#Load model
S = 7
B = 2
C = 3
model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
pretrain_weight = '/home/chaos/Documents/ChaosAIVision/yolo_output/result1/weights/best.pt'

checkpoint = torch.load(pretrain_weight)
model.load_state_dict(checkpoint['model_state_dict'])


#Load image
image = Image.open('/home/chaos/Documents/ChaosAIVision/dataset/yolo_train/valid/images/cat.4069.jpg')
classes =  ["person",'cat', 'dog']


transform = Compose([
                ToTensor(),
                Resize((448,448)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
model.to(DEVICE)
image_transform = transform(image)
image_transform = image_transform.unsqueeze(0)
output = model(image_transform.to(DEVICE))
pred_bboxes = cellboxes_to_boxes(output)
all_pred_boxes = []
batch_size = output.shape[0]
iou_threshold =0.4
conf = 0.40
box_format = 'midpoint'
for idx in range(batch_size):
                    nms_boxes = non_max_suppression(pred_bboxes[idx],
                                                    iou_threshold,
                                                    conf, 
                                                    box_format)
                    for nms_box in nms_boxes:
                        all_pred_boxes.append(nms_box)



#show image
image_np = np.array(image)
# Khởi tạo hình vẽ
fig, ax = plt.subplots(1)

# Hiển thị ảnh
ax.imshow(image_np)

# Thêm bounding boxes
for bbox in all_pred_boxes:
    class_id, conf, x_center, y_center, width, height = bbox
    
    # Chuyển đổi tọa độ YOLO (trung tâm và kích thước) sang (x_min, y_min, width, height)
    x_center *= image_np.shape[1]
    y_center *= image_np.shape[0]
    width *= image_np.shape[1]
    height *= image_np.shape[0]
    
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    
    # Tạo hình chữ nhật (bounding box)
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    # In thông tin lớp và độ tin cậy
    plt.text(x_min, y_min, f'Class: {classes[int(class_id)]} Conf: {conf:.2f}', color='red', fontsize=12, weight='bold', bbox=dict(facecolor='white', alpha=0.7))

# Hiển thị kết quả
plt.axis('off')  # Tắt trục
plt.show()