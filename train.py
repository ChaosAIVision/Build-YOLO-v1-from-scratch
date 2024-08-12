
import os.path
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import argparse
import numpy as np
import shutil
import matplotlib.pyplot as plt
# from utils.general import plot_confusion_matrix
from models.yolo import YOLOv1
from utils.general import ManagerDataYaml, ManageSaveDir, save_plots_from_tensorboard, cellboxes_to_boxes, get_bboxes
from utils.dataloader import CustomDataLoader
from utils.loss import YoloLoss
from utils.dataset import Create_YOLO_Cache
from utils.metric import  non_max_suppression, mean_average_precision,intersection_over_union
import warnings
from torchmetrics.detection.mean_ap import MeanAveragePrecision

warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Train YOLOv1 from scratch")
    parser.add_argument('--data_yaml', "-d", type= str, help= 'Path to dataset', default= '/Users/chaos/Documents/Chaos_working/Chaos_projects/VGG16-from-scratch-Pytorch/data.yaml')
    parser.add_argument('--batch_size', '-b', type = int, help = 'input batch_size')
    parser.add_argument("--image_size", '-i', type = int, default= 448)
    parser.add_argument('--epochs', '-e', type= int, default= 100)
    parser.add_argument('--learning_rate', '-l', type= float, default= 1e-2)
    parser.add_argument('--resume', action='store_true', help='True if want to resume training')
    parser.add_argument('--pretrain', action='store_true', help='True if want to use pre-trained weights')
    return parser.parse_args()  # Cần trả về kết quả từ parser.parse_args()


def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    data_yaml_manage = ManagerDataYaml(args.data_yaml)
    data_yaml_manage.load_yaml()
    pretrain_weight = data_yaml_manage.get_properties(key='pretrain_weight')
    categories = data_yaml_manage.get_properties(key='categories')
    num_classes = data_yaml_manage.get_properties(key='num_classes')
    S= 7, B =2 , C = 20
    model = YOLOv1(split_size= S, num_boxes= B, num_classes= C)
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
    cache_train_creator.__save_cache__()
    if cache_train_creator.__save_cache__() :
        print('sucessfully create train iamges cache !')

    print('Loading valid images ...')
    cache_valid_creator = Create_YOLO_Cache(is_train='valid', data_yaml=args.data_yaml)
    cache_valid_creator.__save_cache__()
    if cache_valid_creator.__save_cache__() :
        print('sucessfully create train iamges cache !')

    model = torch.compile(model)
    model.to(device)
    train_dataloader = CustomDataLoader(args.data_yaml,'train', args.batch_size, num_workers= 4).create_dataloader()
    valid_loader = CustomDataLoader(args.data_yaml,'valid', args.batch_size, num_workers= 2).create_dataloader()
    locate_save_dir = ManageSaveDir(args.data_yaml)
    weights_folder , tensorboard_folder =  locate_save_dir.create_save_dir() # lấy địa chỉ lưu weight và log
    save_dir = locate_save_dir.get_save_dir_path()
    locate_save_dir.plot_dataset() # plot distribution of dataset
    writer = SummaryWriter(tensorboard_folder)
    scaler = torch.cuda.amp.GradScaler()
    # TRAIN
    print(f'result wil save at {save_dir}')
    for epoch in range(start_epochs, args.epochs):
        model.train()
        all_train_losses = []
        all_train_labels = []

        all_train_predictions = []
        progress_bar = tqdm(train_dataloader, colour=  'green', desc=f"epochs: {epoch}/{args.epochs}")
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
                map50 = 0# find my map

                

                
            all_train_losses.append(loss.item())
            all_train_labels.extend(interger_labels.tolist())
            all_train_predictions.extend(prediction_train.tolist())
            progress_bar.set_description(f"Epochs {epoch + 1}/{args.epochs} loss: {loss :0.4f}")
            # writer.add_scalar('Train/loss', loss, epoch * len(train_dataloader) + i)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        avagare__train_loss = np.mean(all_train_losses)
        accuracy_train  = calculate_accuracy(all_train_labels, all_train_predictions, is_all= True)
        cm_train = confusion_matrix(all_train_labels, all_train_predictions)
        precision_recall_train = calculate_precision_recall(cm_train, categories, 'all')
        writer.add_scalar("Train/mean_loss", avagare__train_loss, epoch)
        writer.add_scalar("Train/accuracy", accuracy_train, epoch)
        writer.add_scalar("Train/precision", precision_recall_train['average_precision'], epoch)
        writer.add_scalar("Train/recall", precision_recall_train['average_recall'], epoch)



    
    # VALIDATION

        model.eval()
        all_losses = []
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            progress_bar = tqdm(valid_loader, colour=  'yellow')
            for i, (images, labels) in enumerate(progress_bar):
                images = images.to(device)
                labels = labels.to(device)
                if torch.any(torch.isnan(images)) or torch.any(torch.isinf(images)) or \
                torch.any(torch.isnan(labels)) or torch.any(torch.isinf(labels)):
                    continue
                output = model(images)
                with torch.cuda.amp.autocast():
                    prediction = torch.argmax(output, dim= 1)
                    interger_labels = torch.argmax(labels, dim= 1)
                    if epoch < args.stop_mse_loss: # Use loss MSE when start trainning help model optimizer bettter 
                        output = output.float()
                        labels = labels.float()
                        loss = mse_loss (output, labels)
                    else:
                        loss = CrossEntropyLoss(output, labels)
                progress_bar.set_description(f"Epochs {epoch + 1}/{args.epochs} loss: {loss :0.4f}")
                all_losses.append(loss.item())
                all_labels.extend(interger_labels.tolist())
                all_predictions.extend(prediction.tolist())
                # writer.add_scalar('Valid/loss', loss, epoch * len(valid_loader) + i)


            avagare_loss = np.mean(all_losses)
            accuracy  = calculate_accuracy(all_labels, all_predictions, is_all= True)
            cm = confusion_matrix(all_labels, all_predictions)
            precision_recall = calculate_precision_recall(cm, categories, 'all')
            print(f"precision: {precision_recall['average_precision' ] :0.4f}  recall: {precision_recall['average_recall']:0.4f} loss: {avagare_loss :0.4f} accuracy: {accuracy :0.4f}")
            writer.add_scalar("Valid/accuracy", accuracy, epoch)
            writer.add_scalar("Valid/mean_loss", avagare_loss, epoch)
            writer.add_scalar("Valid/precision", precision_recall['average_precision'], epoch)
            writer.add_scalar("Valid/recall", precision_recall['average_recall'], epoch)
            plot_confusion_matrix(writer, cm, categories, epoch)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epochs' : epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mapuracy': best_map
            }
            torch.save(checkpoint,os.path.join( weights_folder, 'last.pt'))
            if accuracy > best_map:
                torch.save(checkpoint,os.path.join( weights_folder, 'best.pt'))
                best_map = accuracy

    save_plots_from_tensorboard(tensorboard_folder, save_dir)    


        


if __name__ == "__main__":
    args = get_args()
    data_yaml = args.data_yaml
    train(args)