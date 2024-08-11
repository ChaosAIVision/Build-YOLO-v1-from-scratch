import torch
import torch.nn as nn
import pytest

from loss import YOLOloss  # Thay thế `your_module` bằng tên tệp chứa lớp YOLOloss của bạn

# Hàm giả lập để kiểm tra intersection_over_union (bạn cần phải thay thế bằng hàm thực tế của bạn)
def dummy_intersection_over_union(pred_boxes, target_boxes):
    return torch.ones(pred_boxes.size(0), pred_boxes.size(1))

# Đặt lại hàm intersection_over_union để sử dụng hàm giả lập
from utils.metric import intersection_over_union as original_iou
import utils.metric as metric
metric.intersection_over_union = dummy_intersection_over_union

def test_yolo_loss():
    # Khởi tạo mô hình YOLO loss
    yolo_loss = YOLOloss()

    # Tạo các dự đoán giả và mục tiêu giả cho các bài kiểm tra
    batch_size = 2
    S = 7
    B = 2
    C = 20

    # Tạo dự đoán giả (predictions) và mục tiêu giả (target)
    predictions = torch.randn(batch_size, S, S, C + B * 5)
    target = torch.randn(batch_size, S, S, C + B * 5)

    # Tính toán loss
    loss = yolo_loss(predictions, target)

    # Kiểm tra xem loss có phải là giá trị số hay không
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Loss là giá trị scalar

    # Kiểm tra một số điều kiện cụ thể (tùy thuộc vào cách bạn muốn kiểm tra)
    assert loss.item() >= 0  # Loss không thể âm

if __name__ == "__main__":
    pytest.main()
