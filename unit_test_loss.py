import torch
import pytest
from utils.loss import YoloLoss

def create_fake_data(batch_size, S, B, C):
    """
    Tạo dữ liệu giả cho dự đoán và mục tiêu để tính toán loss cho mô hình YOLO.
    
    Parameters:
        batch_size (int): Kích thước của batch.
        S (int): Kích thước lưới.
        B (int): Số lượng hộp dự đoán.
        C (int): Số lượng lớp.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Dự đoán giả và mục tiêu giả.
    """
    # Tạo dự đoán giả (predictions)
    predictions = torch.randn(batch_size, S, S, C + B * 5)

    # Tạo mục tiêu giả (target)
    target = torch.zeros(batch_size, S, S, C + B * 5)
    for i in range(batch_size):
        for j in range(S):
            for k in range(S):
                has_object = torch.randint(0, 2, (1,)).item()
                target[i, j, k, 20] = has_object
                
                if has_object:
                    box = torch.rand(4)  # (x_center, y_center, width, height)
                    class_prob = torch.zeros(C)
                    class_prob[torch.randint(0, C, (1,))] = 1.0
                    
                    target[i, j, k, 21:25] = box  # Các tọa độ của hộp dự đoán
                    target[i, j, k, 25:26] = torch.rand(1)  # Độ tin cậy của hộp chứa đối tượng
                    target[i, j, k, :C] = class_prob  # Xác suất lớp

    return predictions, target

def test_yolo_loss():
    # Khởi tạo mô hình YOLO loss
    yolo_loss = YoloLoss()

    # Tạo các dự đoán giả và mục tiêu giả cho các bài kiểm tra
    batch_size = 2
    S = 7
    B = 2
    C = 20

    # Tạo dự đoán giả (predictions) và mục tiêu giả (target)
    predictions, target = create_fake_data(batch_size, S, B, C)

    # In giá trị để kiểm tra
    print(f"Predictions: {predictions}")
    print(f"Target: {target}")

    # Kiểm tra các giá trị nan
    assert not torch.isnan(predictions).any(), "Predictions contain NaN"
    assert not torch.isnan(target).any(), "Target contains NaN"

    # Tính toán loss
    loss = yolo_loss(predictions, target)
    print(f"Loss: {loss}")

    # Kiểm tra xem loss có phải là giá trị số hay không
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Loss là giá trị scalar

    # Kiểm tra một số điều kiện cụ thể
    assert not torch.isnan(loss).item(), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

if __name__ == "__main__":
    pytest.main()
