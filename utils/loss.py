import torch
import  torch.nn as nn 
from utils.metric import intersection_over_union

class YOLOloss(nn.Module):
    def __init__(self, S= 7, B = 2, C = 20):

        super(YOLOloss).__init__()
        self.mse = nn.MSELoss(reduction= 'sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lamda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.C + self.B*5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[... , 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[... , 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)
        iou_maxes, best_box = torch.max(ious, dim = 0)
        exists_box = target[..., 20].unsqueeze(3)


        # ================================== #
        #        FOR BOX COORDINATES         #
        # ================================== #
        box_predictions = exists_box * (
           ( best_box * predictions[..., 26:30] + (1- best_box) * predictions[..., 21:25])
        )
        # (N, S, S, 4) => (N*S*S, 4)
        box_targets = exists_box * target[...,21:25]
        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(torch.abs(box_predictions[...,2:4 + 1e-6]))
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

        box_loss = self.mse(torch.flatten(box_predictions, end_dim= -2), torch.flatten(box_targets, end_dim= -2),)



        # ================================== #
        #        FOR OBJECT LOSS             #
        # ================================== #


        pred_box =(best_box * predictions[...,25:26] + (1 - best_box) * predictions[...,20:21])
        #(N, S, S)
        object_loss = self.mse(torch.flatten(exists_box * pred_box)), (torch.flatten(exists_box * target[...,20:21]))

        # ================================== #
        #        FOR NO OBJECT LOSS          #
        # ================================== #hhhhhhhhhhhhhhhhhhhgfgfgfhfgh


        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================================== #
        #        FOR CLASS LOSS              #
        # ================================== #


        class_loss = self.mse(torch.flatten(exists_box * predictions[...,20], end_dim= -2), torch.flatten(exists_box*target[...,20], end_dim= -2))

        loss = self.lamda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss

        return loss

