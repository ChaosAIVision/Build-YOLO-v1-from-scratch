
import os.path
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import argparse
import numpy as np
import shutil
import matplotlib.pyplot as plt
from models.yolo import YOLOv1
from utils.general import ManagerDataYaml, ManageSaveDir, save_plots_from_tensorboard, cellboxes_to_boxes, convert_xywh2xyxy
from utils.dataloader import CustomDataLoader
from utils.loss import YoloLoss
from utils.dataset import Create_YOLO_Cache
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from utils.metric import  non_max_suppression
import warnings

warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Train YOLOv1 from scratch")
    parser.add_argument('--data_yaml', "-d", type= str, help= 'Path to dataset', default= '/Users/chaos/Documents/Chaos_working/Chaos_projects/VGG16-from-scratch-Pytorch/data.yaml')
    parser.add_argument('--batch_size', '-b', type = int, help = 'input batch_size')
    parser.add_argument("--image_size", '-i', type = int, default= 448)
    parser.add_argument('--epochs', '-e', type= int, default= 100)
    parser.add_argument('--learning_rate', '-l', type= float, default= 1e-3)
    parser.add_argument('--resume', action='store_true', help='True if want to resume training')
    parser.add_argument('--pretrain', action='store_true', help='True if want to use pre-trained weights')
    parser.add_argument("--iou_threshold", '-iou', type = int, default= 0.5)
    parser.add_argument("--conf_threshold", '-conf', type = int, default= 0.25)

    return parser.parse_args()  # Cần trả về kết quả từ parser.parse_args()


def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    data_yaml_manage = ManagerDataYaml(args.data_yaml)
    data_yaml_manage.load_yaml()
    pretrain_weight = data_yaml_manage.get_properties(key='pretrain_weight')
    categories = data_yaml_manage.get_properties(key='categories')
    num_classes = data_yaml_manage.get_properties(key='num_classes')
    S= 7
    B =2 
    C = 20
    model = YOLOv1(split_size= S, num_boxes= B, num_classes= C)
    # Use DataParallel if more than 1 GPU is available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    loss_fn = YoloLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum= 0.9)
    # optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate, weight_decay=1e-2 )

    best_map = - 100 # create  logic for save weight
    if args.pretrain == True:
        state_dict = torch.load(pretrain_weight)
        model.load_state_dict(state_dict, strict= False)
        print('load weight pretrain sucessfully !')
    if args.resume == True:
        checkpoint = torch.load(pretrain_weight)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epochs = checkpoint['epochs']
        best_map = checkpoint['best_mapuracy']
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print('load weight resume sucessfully !')

    else:
        start_epochs = 0
    #=============
    # LOAD DATASET
    #=============
    print('Loading training images ...')
    cache_train_creator = Create_YOLO_Cache(is_train='train', data_yaml=args.data_yaml)
    if cache_train_creator.__save_cache__() :
        print('sucessfully create train iamges cache !')

    print('Loading valid images ...')
    cache_valid_creator = Create_YOLO_Cache(is_train='valid', data_yaml=args.data_yaml)
    if cache_valid_creator.__save_cache__() :
        print('sucessfully create train iamges cache !')

    model = torch.compile(model)
    model.to(device)
    train_dataloader = CustomDataLoader(args.data_yaml,'train', args.batch_size, num_workers= 4).create_dataloader()
    valid_loader = CustomDataLoader(args.data_yaml,'valid', args.batch_size, num_workers= 2).create_dataloader()
    locate_save_dir = ManageSaveDir(args.data_yaml)
    weights_folder , tensorboard_folder =  locate_save_dir.create_save_dir() # lấy địa chỉ lưu weight và log
    save_dir = locate_save_dir.get_save_dir_path()
    # locate_save_dir.plot_dataset() # plot distribution of dataset
    writer = SummaryWriter(tensorboard_folder)
    scaler = torch.cuda.amp.GradScaler()
    # TRAIN
    print(f'result wil save at {save_dir}')
    for epoch in range(start_epochs, args.epochs):
        model.train()
        all_train_losses = []
        progress_bar = tqdm(train_dataloader, colour=  'green', desc=f"epochs: {epoch + 1}/{args.epochs}")
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            if torch.any(torch.isnan(images)) or torch.any(torch.isinf(images)) or \
           torch.any(torch.isnan(labels)) or torch.any(torch.isinf(labels)):
                continue
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output =  model(images)
                loss = loss_fn(output, labels)
                all_train_losses.append(loss.item())                
            all_train_losses.append(loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():0.4f}'})

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Thay đổi max_norm tùy thuộc vào nhu cầu của bạn
            scaler.step(optimizer)
            scaler.update()
        avagare__train_loss = np.mean(all_train_losses)
        writer.add_scalar("Train/mean_loss", avagare__train_loss, epoch)



    
    # VALIDATION

        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")
        iou_threshold = args.iou_threshold
        conf = args.conf_threshold
        pred_format ='cells'
        box_format = 'midpoint'
        all_losses = []
        all_pred_boxes = []
        all_true_boxes = []
  

        with torch.no_grad():
            progress_bar = tqdm(valid_loader, colour=  'yellow', desc=f"epochs: {epoch +1}/{args.epochs}")
            for i, (images, labels) in enumerate(progress_bar):
                images = images.to(device)
                labels = labels.to(device)
                if torch.any(torch.isnan(images)) or torch.any(torch.isinf(images)) or \
                torch.any(torch.isnan(labels)) or torch.any(torch.isinf(labels)):
                    continue
                output = model(images)
                with torch.cuda.amp.autocast():
                    loss = loss_fn(output, labels)
                all_losses.append(loss.item())
                progress_bar.set_postfix({'loss': f'{loss.item():0.4f}'})
                batch_size = output.shape[0]
                true_bboxes =cellboxes_to_boxes(labels)
                pred_bboxes = cellboxes_to_boxes(output)
                for idx in range(batch_size):

                    nms_boxes = non_max_suppression(pred_bboxes[idx],
                                                    iou_threshold,
                                                    conf, 
                                                    box_format)
                    for nms_box in nms_boxes:
                        all_pred_boxes.append(nms_box)
                    for box in true_bboxes[idx]:
                        if box[1] > conf:
                            all_true_boxes.append(box)
            if len(all_pred_boxes) == 0:
                mAP50= 0
            else:
                true_boxes_tensor, true_labels_tensor, _ = convert_xywh2xyxy(all_true_boxes, 448)
                pred_boxes_tensor, pred_labels_tensor, pred_scores_tensor = convert_xywh2xyxy(all_pred_boxes, 448)
                true_boxes = [{"boxes": true_boxes_tensor, "labels": true_labels_tensor}]
                pred_boxes = [{"boxes": pred_boxes_tensor, "scores": pred_scores_tensor, "labels": pred_labels_tensor}]
                metric.update(pred_boxes, true_boxes)
                result = metric.compute()
                mAP50 = result['map_50']
            avagare_loss = np.mean(all_losses)
            print(f"mAP50: {mAP50 :0.4f}, mean_loss: {avagare_loss: 0.4f} ")
            writer.add_scalar("Valid/mAP50", mAP50, epoch)
            writer.add_scalar("Valid/mean_loss", avagare_loss, epoch)
          
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epochs' : epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mapuracy': best_map
            }
            torch.save(checkpoint,os.path.join( weights_folder, 'last.pt'))
            if mAP50 > best_map:
                torch.save(checkpoint,os.path.join( weights_folder, 'best.pt'))
                best_map = mAP50

    save_plots_from_tensorboard(tensorboard_folder, save_dir)    


        


if __name__ == "__main__":
    args = get_args()
    data_yaml = args.data_yaml
    train(args)