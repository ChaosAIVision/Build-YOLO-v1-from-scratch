import pandas as pd
import shutil
import os

def copy_files_from_excel(image_folder, label_folder, excel_path, output_folder):
    # Đọc tệp Excel
    df = pd.read_csv(excel_path)
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels'), exist_ok=True)
    
    for index, row in df.iterrows():
        image_file = row['image']
        label_file = row['text']
        
        # Tạo đường dẫn đầy đủ đến các tệp nguồn
        image_source_path = os.path.join(image_folder, image_file)
        label_source_path = os.path.join(label_folder, label_file)
        
        # Tạo đường dẫn đích
        image_dest_path = os.path.join(output_folder, 'images', image_file)
        label_dest_path = os.path.join(output_folder, 'labels', label_file)
        
        # Sao chép tệp ảnh nếu tồn tại
        if os.path.exists(image_source_path):
            shutil.copy(image_source_path, image_dest_path)
        else:
            print(f"Ảnh không tồn tại: {image_source_path}")
        
        # Sao chép tệp nhãn nếu tồn tại
        if os.path.exists(label_source_path):
            shutil.copy(label_source_path, label_dest_path)
        else:
            print(f"Nhãn không tồn tại: {label_source_path}")

# Ví dụ gọi hàm:
copy_files_from_excel(
    image_folder='/home/chaos/Downloads/images',
    label_folder='/home/chaos/Downloads/labels',
    excel_path='/home/chaos/Downloads/100examples.csv',
    output_folder='/home/chaos/Documents/ChaosAIVision/Build-YOLO-v1-from-scratch/dataset'
)
