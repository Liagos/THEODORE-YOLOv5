import os
import torch
# import wandb
import configparser
import pandas as pd
from tqdm import tqdm
from loss import YoloLoss
import torch.optim as optim
from YOLO_V5_V2 import YOLOV5, ScalePrediction
from YOLOV5_Torch import YoloV5
from torchsummary import summary
from numpy.random import RandomState
from config_layers import ANCHORS, S
from YOLO_Dataset import YOLODataset
from torch.utils.data import DataLoader
from YOLO_functions import getEvaluationBoxes, mean_average_precision, check_class_accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = True

parser = configparser.ConfigParser()
parser.read("config.ini")
outputPath = parser.get("outputPath", "output_path")
num_classes = len(parser.get("classNames", "names").split(", "))


def train_fn(model, optimizer, scaler, train_loader, loss_fn, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch, (x, y) in enumerate(loop):
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )

        with torch.cuda.amp.autocast():
            out = model(x)

            loss = (loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2]))
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        # wandb.log({"loss": loss})

        mean_loss = sum(losses) / len(losses)

        loop.set_postfix(loss=mean_loss)

    return mean_loss


def main(nepochs, batchSize, lr, weight_decay):
    model = YOLOV5(in_channels=3, num_classes=num_classes).to(device)
    # model.load_state_dict(torch.load("myModel - Copy.pt")) #  load parameters
    # for param in model.parameters(): #  freeze all layers except the head
    #     param.requires_grad = False
    # for child in list(model.children()):
    #     for param in list(child.children()):
    #         if isinstance(param, ScalePrediction):
    #             for p in param.parameters():
    #                 p.requires_grad = True

    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.937))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.937, weight_decay=weight_decay, nesterov=True)
    loss_fn = YoloLoss()

    scaler = torch.cuda.amp.GradScaler()

    # wandb.init(project="my-test-project")

    # wandb.config = {
    #     "learning_rate": lr,
    #     "epochs": nepochs,
    #     "batch_size": 1
    # }

    dataset = pd.read_csv(os.path.join(outputPath + "/dataset.csv"))
    df = dataset.sample(n=dataset.shape[0], random_state=RandomState())
    train = df.sample(frac=0.8, random_state=RandomState())
    test = df.loc[~df.index.isin(train.index)]
    train.to_csv(outputPath + "/train.csv", index=False, header=False)
    test.to_csv(outputPath + "/test.csv", index=False, header=False)

    train_dataset = YOLODataset(outputPath + "/train.csv",
                                anchors=ANCHORS)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batchSize,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=False,
                              shuffle=True)

    test_dataset = YOLODataset(outputPath + "/test.csv",
                               anchors=ANCHORS)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batchSize,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)

    scaled_anchors = (
            torch.tensor(ANCHORS) * torch.tensor(S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)).to(device)

    for epoch in range(nepochs):
        print(f"Epoch: ", epoch)

        train_fn(model, optimizer, scaler, train_loader, loss_fn, scaled_anchors)
        if epoch % 9 == 0 and epoch > 0:
            check_class_accuracy(model, test_loader, threshold=0.05)
            pred_boxes, true_boxes = getEvaluationBoxes(
                model,
                test_loader,
                iou_threshold=0.6,
                anchors=ANCHORS,
                threshold=0.05,
            )
            mapval, TP_cumsum, FP_cumsum, recalls, precisions = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=0.6,
                num_classes=num_classes,
            )
            print(f"MAP: {mapval.item()}")
            print(f"True Positives: {TP_cumsum[-1].item()}")
            print(f"False Positives: {FP_cumsum[-1].item()}")
            print(f"Recall: {round(recalls[-1].item(), 2)}")
            print(f"Precision: {round(precisions[-1].item(), 2)}")

    return model, pred_boxes, true_boxes


if __name__ == "__main__":
    # model = YOLOV5(in_channels=3, num_classes=num_classes).to(device)
    # summary(model, [3,416,416])
    model, pred_boxes, true_boxes = main(nepochs=1, batchSize=32, lr=0.01, weight_decay=0.0005)
    import pickle

    a_file = open("/content/drive/MyDrive/THEODORE/pred_boxes.pkl", "wb")
    pickle.dump(pred_boxes, a_file)
    a_file.close()

    b_file = open("/content/drive/MyDrive/THEODORE/true_boxes.pkl", "wb")
    pickle.dump(true_boxes, b_file)
    b_file.close()
