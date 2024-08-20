
import os.path
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import argparse
import numpy as np
import shutil
import matplotlib.pyplot as plt
from models.yolo import Yolov1
from utils.general import ManagerDataYaml, ManageSaveDir, save_plots_from_tensorboard, cellboxes_to_boxes, convert_xywh2xyxy, rename_keys
from utils.dataloader import CustomDataLoader
from utils.loss import YoloLoss
from utils.dataset import Create_YOLO_Cache
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from termcolor import colored


from utils.metric import  non_max_suppression,mean_average_precision
import warnings

warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Train YOLOv1 from scratch")
    parser.add_argument('--data_yaml', "-d", type= str, help= 'Path to dataset', default= '/Users/chaos/Documents/Chaos_working/Chaos_projects/VGG16-from-scratch-Pytorch/data.yaml')
    parser.add_argument('--batch_size', '-b', type = int, help = 'input batch_size')
    parser.add_argument("--image_size", '-i', type = int, default= 448)
    parser.add_argument('--epochs', '-e', type= int, default= 100)
    parser.add_argument('--learning_rate', '-l', type= float, default= 2e-5)
    parser.add_argument('--resume', action='store_true', help='True if want to resume training')
    parser.add_argument('--pretrain', action='store_true', help='True if want to use pre-trained weights')
    parser.add_argument("--iou_threshold", '-iou', type = int, default= 0.5)
    parser.add_argument("--conf_threshold", '-conf', type = int, default= 0.1)

    return parser.parse_args()  # Cần trả về kết quả từ parser.parse_args()


def get_bboxes_training(
    outputs,
    labels,
    iou_threshold=0.5,
    threshold=0.4,
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # Ensure the model is in evaluation mode before obtaining bounding boxes
    train_idx = 0

    true_bboxes = cellboxes_to_boxes(labels)
    bboxes = cellboxes_to_boxes(outputs)

    for idx in range(outputs.shape[0]):
        nms_boxes = non_max_suppression(
            bboxes[idx],
            iou_threshold=iou_threshold,
            threshold=threshold,
            box_format=box_format,
        )

        for nms_box in nms_boxes:
            all_pred_boxes.append([train_idx] + nms_box)

        for box in true_bboxes[idx]:
            # Convert multiple boxes to 0 if predicted
            if box[1] > threshold:
                all_true_boxes.append([train_idx] + box)

        train_idx += 1

    return all_pred_boxes, all_true_boxes

def train_fn(train_loader, model, optimizer, loss_fn, epoch, total_epochs):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    mean_loss = []
    mean_mAP = []

    # Dùng `leave=True` để đảm bảo tqdm hoàn thành thanh tiến trình đúng cách
    progress_bar = tqdm(train_loader, colour='green', desc=f"Epochs: {epoch + 1}/{total_epochs}", leave=True)

    for batch_idx, (x, y) in enumerate(progress_bar):
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)) or \
            torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
                continue
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        pred_boxes, true_boxes = get_bboxes_training(out, y, iou_threshold=0.5, threshold=0.4)
        mAP = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint")
        progress_bar.set_postfix({'loss': f'{loss.item():0.4f}'})

        mean_loss.append(loss.item())
        mean_mAP.append(mAP.item())

    avg_loss = sum(mean_loss) / len(mean_loss)
    avg_mAP = sum(mean_mAP) / len(mean_mAP)
    print(colored(f"\nTrain \t loss: {avg_loss:3.10f} \t mAP: {avg_mAP:3.10f}\n", 'green'))

    return avg_mAP, avg_loss

def test_fn(test_loader, model, loss_fn, epoch, total_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    model.eval()
    mean_loss = []
    mean_mAP = []

    progress_bar = tqdm(test_loader, colour='yellow', desc=f"Epochs: {epoch + 1}/{total_epochs}", leave=True)

    for batch_idx, (x, y) in enumerate(progress_bar):
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)) or \
            torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
                continue
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)

        pred_boxes, true_boxes = get_bboxes_training(out, y, iou_threshold=0.5, threshold=0.4)
        mAP = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint")

        progress_bar.set_postfix({'loss': f'{loss.item():0.4f}'})
        mean_loss.append(loss.item())
        mean_mAP.append(mAP.item())

    avg_loss = sum(mean_loss) / len(mean_loss)
    avg_mAP = sum(mean_mAP) / len(mean_mAP)

    print(colored(f"\nTest \t loss: {avg_loss:3.10f} \t mAP: {avg_mAP:3.10f}\n", 'yellow'))

    return avg_mAP, avg_loss

def train(args):
    seed = 123
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    data_yaml_manage = ManagerDataYaml(args.data_yaml)
    data_yaml_manage.load_yaml()
    pretrain_weight = data_yaml_manage.get_properties(key='pretrain_weight')
    categories = data_yaml_manage.get_properties(key='categories')
    num_classes = data_yaml_manage.get_properties(key='num_classes')
    S = 7
    B = 2
    C = 3
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    ##############################################################################################

    pretrain_weight = '/home/chaos/Documents/ChaosAIVision/temp_folder/backbone448/weights/last.pt'
    checkpoint = torch.load(pretrain_weight)
    backbone_state_dict = rename_keys(checkpoint['model_state_dict'])


   

    # model = torch.compile(model)
  
    model.darknet.load_state_dict(backbone_state_dict, strict=False)
    print('Loading backbone pretrain successfully !')
    # for param in model.darknet.parameters():
    #     param.requires_grad = False
    #################################################################################################
    # # Kiểm tra xem các tham số của backbone đã được đóng băng chưa
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    
    # Use DataParallel if more than 1 GPU is available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    loss_fn = YoloLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum= 0.9)

    if args.pretrain:
        state_dict = torch.load(pretrain_weight)
        model.load_state_dict(state_dict, strict=False)
        print('Loaded pretrain weights successfully!')
    
  
    # Load dataset
    print('Loading training images ...')
    cache_train_creator = Create_YOLO_Cache(is_train='train', data_yaml=args.data_yaml)
    if cache_train_creator.__save_cache__():
        print('Successfully created train images cache!')

    print('Loading valid images ...')
    cache_valid_creator = Create_YOLO_Cache(is_train='valid', data_yaml=args.data_yaml)
    if cache_valid_creator.__save_cache__():
        print('Successfully created valid images cache!')

    model.to(device)
    train_dataloader = CustomDataLoader(args.data_yaml, 'train', args.batch_size, num_workers=4).create_dataloader()
    valid_loader = CustomDataLoader(args.data_yaml, 'valid', args.batch_size, num_workers=2).create_dataloader()
    locate_save_dir = ManageSaveDir(args.data_yaml)
    weights_folder, tensorboard_folder = locate_save_dir.create_save_dir()  # Get save directories for weights and logs
    save_dir = locate_save_dir.get_save_dir_path()
    writer = SummaryWriter(tensorboard_folder)

    #TRAIN
    print(f'Results will be saved at {save_dir}')

    best_mAP_train = 0
    best_mAP_test = 0

    for epoch in range(args.epochs):
        train_mAP, train_avg_loss = train_fn(train_dataloader, model, optimizer, loss_fn, epoch, args.epochs)
        valid_mAP, valid_avg_loss = test_fn(valid_loader, model, loss_fn, epoch, args.epochs)
        # Write mAP and meanLoss to plot
        writer.add_scalar("Train/mAP50", train_mAP, epoch)
        writer.add_scalar("Train/mean_loss", train_avg_loss, epoch)
        writer.add_scalar("Valid/mAP50", valid_mAP, epoch)
        writer.add_scalar("Valid/mean_loss", valid_avg_loss, epoch)

        checkpoint = {
            'model_state_dict': model.state_dict()}
        torch.save(checkpoint, os.path.join(weights_folder, 'last.pt'))


        # Update the best mAP for train and test
        if train_mAP > best_mAP_train:
            best_mAP_train = train_mAP

        if valid_mAP > best_mAP_test:
            best_mAP_test = valid_mAP
            torch.save(checkpoint, os.path.join(weights_folder, 'best.pt'))


    print(colored(f"Best Train mAP: {best_mAP_train:3.10f}", 'green'))
    print(colored(f"Best Test mAP: {best_mAP_test:3.10f}", 'yellow'))
 
    save_plots_from_tensorboard(tensorboard_folder, save_dir)


        


if __name__ == "__main__":
    args = get_args()
    data_yaml = args.data_yaml
    train(args)