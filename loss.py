import torch
import torch.nn as nn
from YOLO_functions import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, targets, anchors):
        obj = targets[..., 0] == 1
        noobj = targets[..., 0] == 0

        # No Object Loss!
        no_object_loss = self.bce((predictions[..., 0:1][noobj]), (targets[..., 0:1][noobj]))

        # Object Loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)  # incoming anchors shape 3 x 2, that's why we reshape, p_w * exp(t_w)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)  # sigmoid on x and y coordinate, and exp on w and h
        ious = intersection_over_union(box_preds[obj], targets[..., 1:5][obj]).detach()
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * targets[..., 0:1][obj]))

        # Box Coordinate Loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x and y between 0,1
        targets[..., 3:5] = torch.log(1e-16 + targets[..., 3:5] / anchors)
        box_loss = self.mse(predictions[..., 1:5][obj], targets[..., 1:5][obj])

        # Class Loss
        class_loss = self.entropy((predictions[..., 5:][obj]), (targets[..., 5][obj].long()))

        return (self.lambda_box * box_loss +
                self.lambda_obj * object_loss +
                self.lambda_noobj * no_object_loss +
                self.lambda_class * class_loss)
