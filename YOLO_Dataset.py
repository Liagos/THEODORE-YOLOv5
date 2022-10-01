import os
import torch
import numpy as np
import configparser
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import torch
from args import Arguments
from config_layers import ANCHORS
# from torchvision.transforms import ToTensor
from albumentations.pytorch import ToTensorV2
from YOLO_functions import IoU_width_height, cells_to_bboxes, non_maximum_suppression, plot_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

parser = configparser.ConfigParser()
parser.read("config.ini")
classNames = parser.get("classNames", "names")
classNames = classNames.split(", ")

class YOLODataset(Dataset):
    def __init__(self, csv_file, anchors, S=[52, 26, 13], C=len(classNames)):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_threshold = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_dir = "/content/drive/MyDrive/Dataset/labels"
        image_dir = "/content/drive/MyDrive/Dataset/images"
        label_path = os.path.join(label_dir, self.annotations.iloc[index, 3])
        bboxes = np.loadtxt(fname=label_path, delimiter=",", ndmin=2).tolist()
        img_path = os.path.join(image_dir, self.annotations.iloc[index, 1])
        image = np.array(Image.open(img_path).resize((416, 416), resample=2))
        targets = [torch.zeros(self.num_anchors_per_scale, S, S, 6) for S in self.S]  # [p_o, x, y, w, h, c]

        for box in bboxes:
            iou = IoU_width_height(torch.tensor(box[2:4]), self.anchors)  # Calculate IoU with width and height
            anchor_indices = iou.argsort(descending=True, dim=0)
            class_label, x, y, width, height = box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode="trunc")  # Will give row
                anchor_of_scale = anchor_idx % self.num_anchors_per_scale  # Will give column or (index - (row * width of list))
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_of_scale, i, j, 0]
                if not has_anchor[scale_idx] and not anchor_taken:
                    targets[scale_idx][anchor_of_scale, i, j, 0] = 1
                    cell_x, cell_y = S * x - j, S * y - i
                    width_cell, height_cell = (S * width, S * height)
                    box_coordinates = torch.tensor([cell_x, cell_y, width_cell, height_cell])
                    targets[scale_idx][anchor_of_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_of_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou[anchor_idx] > self.ignore_iou_threshold:
                    targets[scale_idx][anchor_of_scale, i, j, 0] = -1  # ignore prediction

        norm = A.Compose([A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
                          ToTensorV2()])
        aug = norm(image=image)
        image = aug["image"]

        return image, tuple(targets)


# def test():
#     parser = configparser.ConfigParser()
#     parser.read("config.ini")
#     outputPath = parser.get("outputPath", "output_path")
#     # trainPath = parser.get("trainImageDir", "path")
#     # trainLabel = parser.get("trainLabelDir", "path")
#
#     anchors = ANCHORS
#
#     dataset = YOLODataset(outputPath + "/dataset.csv",
#                           S=[52, 26, 13],
#                           anchors=anchors,
#                           C=len(classNames))
#
#     S = [52, 26, 13]
#     scaled_anchors = torch.tensor(anchors) / (
#             1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
#     )
#     loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
#     for x, y in loader:
#         boxes = []
#         for i in range(y[0].shape[1]):
#             anchor = scaled_anchors[i]
#             boxes += cells_to_bboxes(
#                 y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
#             )[0]
#         boxes = non_maximum_suppression(boxes, iou_threshold=1, threshold=0.7)
#         plot_image(x[0].to("cpu"), boxes)


# if __name__ == "__main__":
#     test()
