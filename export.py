from tqdm import tqdm
import numpy as np

# Giả sử bạn có một danh sách lớn các dữ liệu
data = np.random.rand(10000, 10)  # Ví dụ danh sách dữ liệu

# Khởi tạo thanh tiến trình
epochs = 0
pbar = tqdm(range(len(data)), desc=f"epochs: {epochs}")

for i in pbar:
    # Xử lý dữ liệu
    processed_data = np.mean(data[i])  # Ví dụ xử lý dữ liệu
    
    # Cập nhật thông tin bổ sung trong mỗi bước của vòng lặp
    pbar.set_postfix({ 'ngu': ' ngu', 'sucvat': i, 'mean': processed_data})
    epochs += 1
