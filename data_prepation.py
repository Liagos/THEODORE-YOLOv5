import os
import csv
import cv2
import time
import shutil
import numpy as np
import configparser
import pandas as pd
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import tensorflow as tf
from args import Arguments
from collections import Counter

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_feature(string_record):
    feature = {
        'image/height': tf.io.VarLenFeature(tf.int64),
        'image/width': tf.io.VarLenFeature(tf.int64),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(string_record, feature)

    return example


def iterateDataset(tfrecordFile, keep_class=None):
    ds = tf.data.TFRecordDataset(tfrecordFile)
    ds = ds.map(parse_feature, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    image_list = []
    true_bboxes = []
    class_with_label = []

    for i, example in enumerate(tqdm(ds), 5):
        coordinates = []
        height = int(example['image/height'].values)
        width = int(example['image/width'].values)

        img_id = (example['image/source_id'].numpy())
        key = example['image/key/sha256'].numpy()

        img_string = (example['image/encoded'].numpy())
        img_filename = (example['image/filename'].numpy()).decode('utf-8')

        pil_img = Image.open(BytesIO(img_string))
        img = np.array(pil_img)

        bb_xmin = np.array(example['image/object/bbox/xmin'].values)
        bb_ymin = np.array(example['image/object/bbox/ymin'].values)
        bb_xmax = np.array(example['image/object/bbox/xmax'].values)
        bb_ymax = np.array(example['image/object/bbox/ymax'].values)
        classes = np.array(example['image/object/class/text'].values)
        labels = np.array(example['image/object/class/label'].values)

        classes = [str(c).replace("b", "", 1) for c in classes]  # remove starting letter b
        classes = [str(c).replace("'", "") for c in classes]  # remove hyphens

        for xmin, ymin, xmax, ymax, label, cls in zip(bb_xmin, bb_ymin, bb_xmax, bb_ymax, labels, classes):
            if keep_class:
                if keep_class == cls:
                    x = int(round(xmin * width))
                    y = int(round(ymin * height))
                    x2 = int(round(xmax * width))
                    y2 = int(round(ymax * height))
                    coordinates.append([label-1, x, y, x2 - x, y2 - y])  # Append label,x,y,w,h
                    class_with_label.append((cls, label-1))  # Class along with corresponding label
            else:
                x = int(round(xmin * width))
                y = int(round(ymin * height))
                x2 = int(round(xmax * width))
                y2 = int(round(ymax * height))
                coordinates.append([label - 1, x, y, x2 - x, y2 - y])  # Append label,x,y,w,h
                class_with_label.append((cls, label - 1))  # Class along with corresponding label

        image_list.append(img)  # List of all images
        for c in coordinates:
            true_bboxes.append([i] + c)  # bboxes for corresponding image
        # if i == 10:
        #     break

    return image_list, true_bboxes, class_with_label


def sortClasses(classAndLabel):
    class_w_label_freq = Counter(classAndLabel)  # Extract classes with labels and frequency
    class_names = []
    class_labels = []
    freq = []
    for key, value in class_w_label_freq.items():
        class_names.append(key[0])
        class_labels.append(key[1])
        freq.append(value)

    sorted_labels, sorted_names, frequency = map(list, zip(*sorted(zip(class_labels, class_names, freq), reverse=False)))

    return sorted_names


def convert2YOLO(boundingBoxes, image):
    yolo_box_list = []
    h, w = image.shape[:2]
    for box in boundingBoxes:
        x = (box[2] + box[4] / 2.0) / w
        y = (box[3] + box[5] / 2.0) / h
        width = box[4] / w
        height = box[5] / h
        yolo_box_list.append((box[1], x, y, width, height))
    return yolo_box_list


def saveFiles(path, imgList, boundingBox):
    imageNames = []
    labelNames = []
    saveImagePathList = []
    saveLabelPathList = []
    if os.path.exists(path):
        print('Output folder already exists: {}'.format(path))
        print('Do you want to delete this output folder?: [y]')
        inp = input()
        if inp == 'y':
            shutil.rmtree(path)
            time.sleep(0.5)
        else:
            print('Script finished')
            exit()
    os.mkdir(path)
    subFolders = ['images', 'labels']
    for sub in subFolders:
        os.mkdir(str(path) + "/" + str(sub))
    for idx, image in enumerate(tqdm(imgList), 5):
        boxes_to_save = [box for box in boundingBox if box[0] == idx]
        saveImagePath = os.path.join(path, subFolders[0], "image_{}.jpeg".format(str(idx).zfill(6)))
        saveImagePathList.append(os.path.join(path, subFolders[0]))
        imageNames.append("image_{}.jpeg".format(str(idx).zfill(6)))
        saveLabelPath = os.path.join(path, subFolders[1])
        saveLabelPathList.append(saveLabelPath)
        cv2.imwrite(saveImagePath, image)
        with open(saveLabelPath + "/" + "image_{}.txt".format(str(idx).zfill(6)), mode="w", newline='') as f:
            writer = csv.writer(f, delimiter=",")
            labelNames.append("image_{}.txt".format(str(idx).zfill(6)))
            yolo_bboxes = convert2YOLO(boxes_to_save, image)
            for box in yolo_bboxes:
                writer.writerow((int(box[0]), box[1], box[2], box[3], box[4]))
        df = pd.DataFrame(data=(saveImagePathList, imageNames, saveLabelPathList, labelNames)).T
        df.to_csv(path + "/" + "dataset.csv", index=False, header=False)


def updateConfigFile(class_names_list):
    configFile = "config.ini"
    config_file = configparser.ConfigParser()
    config_file.read(configFile)

    class_names_list = ", ".join(class_names_list)

    if not config_file.has_section("classNames"):
        config_file.add_section("classNames")
    config_file.set("classNames", 'names', class_names_list)

    with open('config.ini', 'w') as configfile:
        config_file.write(configfile)


if __name__ == "__main__":
    arguments = Arguments()
    parser = arguments.getArgs()
    outputPath = parser.outputPath
    tfRecordFile = "training_50k.tfrecord"
    examples = parse_feature
    imageList, bboxes, classLabel = iterateDataset(tfRecordFile, keep_class='person')
    sortedClasses = sortClasses(classLabel)
    updateConfigFile(sortedClasses)
    saveFiles(outputPath, imageList, bboxes)
