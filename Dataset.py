import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from lxml.etree import Element, SubElement, ElementTree
import json
import shutil
from shutil import copyfile

SEED = 2021

train_audio_path = "./train"
test_audio_path = "./test"
label_path = "./label"

train_tp = "./train_tp.csv"
train_fp = "./train_fp.csv"
sample_submission = "./sample_submission.csv"

train_data_tp = pd.read_csv(train_tp)
train_data_fp = pd.read_csv(train_fp)

imglist = os.listdir("./train_tran")

data = train_data_tp[["recording_id", "species_id"]]
label_data = pd.read_table("./label/label.txt", sep="\t")

Label = [[], [], []]

for img in tqdm(imglist):
    cur_img = "./train_tran/" + img
    data = label_data.loc[label_data["location"] == cur_img][["t_min", "t_max", "f_min", "f_max", "species_id",
                                                              "songtype_id"]]
    Label[0].append(img)
    # 将data直接打包，方便接下来生成xml格式的标记
    Label[1].append(data)
    # 每一张图片的主要label作为该图的label，用于分层采样
    Label[2].append(int(np.array(data["species_id"])[0]))
# 按照6:2:2分配训练集、验证集、测试集
X_train_validate, X_test, y_train_validate, y_test = train_test_split(Label[0], Label[1],
                                                                      stratify=Label[2],
                                                                      test_size=0.2, random_state=SEED)
Label_list = [int(np.array(label)[0][4]) for label in y_train_validate]
X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate, y_train_validate, test_size=0.25,
                                                            stratify=Label_list, random_state=SEED)

print("训练集数量: ", len(X_train))
print("验证集数量: ", len(X_validate))
print("测试集数量: ", len(X_test))
# print(len(y_validate))
# print(len(y_train))
# print(len(y_test))

# 创建coco所需要的文件夹
if not os.path.exists('./coco/'):
    os.makedirs('./coco/')
if not os.path.exists('./coco/annotations/'):
    os.makedirs('./coco/annotations/')
if not os.path.exists('./coco/images/train2017/'):
    os.makedirs('./coco/images/train2017/')
if not os.path.exists('./coco/images/val2017/'):
    os.makedirs('./coco/images/val2017/')

# 创建VOC所需要的文件夹
if not os.path.exists('./VOC2007/'):
    os.makedirs('./VOC2007/')
if not os.path.exists('./VOC2007/JPEGImages/'):
    os.makedirs('./VOC2007/JPEGImages/')
if not os.path.exists('./VOC2007/Annotations'):
    os.makedirs('./VOC2007/Annotations')
if not os.path.exists('./VOC2007/ImageSets'):
    os.makedirs('./VOC2007/ImageSets')

xml_location = "./VOC2007/Annotations/"
# 制作VOC格式的label
# warning: VOC格式中的种类从1开始
for i in range(len(Label[0])):
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC2007'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = Label[0][i]
    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = "Kaggle Rainforest Connection Species Audio Detection Database"
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '387'
    node_height = SubElement(node_size, 'height')
    node_height.text = '154'
    # 图像通道数
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    # segmented表示是否具有分割数据
    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'
    for j in range(Label[1][i].shape[0]):
        # xml file generator
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        # 0默认为背景
        node_name.text = str(int(np.array(Label[1][i]["species_id"])[j]) + 1)
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        # 在VOC2007 Dataset中, 坐标系与图像坐标系相同，以左上角为圆点，垂直向下为y轴，水平向右为x轴
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(np.array(Label[1][i]["t_min"])[j])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(154 - np.array(Label[1][i]["f_max"])[j])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(np.array(Label[1][i]["t_max"])[j])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(154 - np.array(Label[1][i]["f_min"])[j])
    doc = ElementTree(node_root)
    (filename, extension) = os.path.splitext(Label[0][i])
    # xml file writer
    # pretty_print 正确换行
    # xml_declaration False 去除xml文件声明
    doc.write(open(xml_location + filename + ".xml", "wb"), encoding="utf-8", xml_declaration=False, pretty_print=True)
print("VOC Annotations xml files generate successfully!")
# 制作VOC格式的train.txt val.txt trainval.txt test.txt
# VOC格式要求第一列为文件名（去除后缀名）, 第二列为±1, 1表示正样本, -1表示负样本
file = open('./VOC2007/ImageSets/Main/trainval.txt', 'w')
for i in range(len(X_train_validate)):
    (filename, extension) = os.path.splitext(X_train_validate[i])
    file.write(filename + "\t" + "1" + "\n")
file.close()
print("VOC trainval.txt generate successfully!")
file = open('./VOC2007/ImageSets/Main/train.txt', 'w')
for i in range(len(X_train)):
    (filename, extension) = os.path.splitext(X_train[i])
    file.write(filename + "\t" + "1" + "\n")
file.close()
print("VOC train.txt generate successfully!")
file = open('./VOC2007/ImageSets/Main/val.txt', 'w')
for i in range(len(X_validate)):
    (filename, extension) = os.path.splitext(X_validate[i])
    file.write(filename + "\t" + "1" + "\n")
file.close()
print("VOC val.txt generate successfully!")
file = open('./VOC2007/ImageSets/Main/test.txt', 'w')
for i in range(len(X_test)):
    (filename, extension) = os.path.splitext(X_test[i])
    file.write(filename + "\t" + "1" + "\n")
file.close()
print("VOC test.txt generate successfully!")

img_dataset = "./train_img/"
# 每次把图片文件夹清空
if len(os.listdir("./coco/images/train2017/")):
    shutil.rmtree("./coco/images/train2017/")
    os.mkdir("./coco/images/train2017/")
if len(os.listdir("./coco/images/val2017/")):
    shutil.rmtree("./coco/images/val2017/")
    os.mkdir("./coco/images/val2017/")
if len(os.listdir("./coco/images/test2017/")):
    shutil.rmtree("./coco/images/test2017/")
    os.mkdir("./coco/images/test2017/")

# 制作COCO格式的json
coco_dataset = {'categories': [], 'images': [], 'annotations': []}
imgcount = 0
bdboxcount = 0
for i in range(len(X_train)):
    copyfile(img_dataset + X_train[i], "./coco/images/train2017/" + X_train[i])
    coco_dataset['images'].append({'file_name': X_train[i],
                                   'id': imgcount,
                                   'width': 387,
                                   'height': 154})
    for j in range(y_train[i].shape[0]):
        coco_dataset['annotations'].append({
            'area': (np.array(y_train[i]["t_max"])[j] - np.array(y_train[i]["t_min"])[j]) * (
                    np.array(y_train[i]["f_max"])[j] - np.array(y_train[i]["f_min"])[j]),
            # x1, y1为左上角顶点
            # bbox: x1, y1, width, height
            'bbox': [np.array(y_train[i]["t_min"])[j], 154 - np.array(y_train[i]["f_max"])[j],
                     np.array(y_train[i]["t_max"])[j] - np.array(y_train[i]["t_min"])[j],
                     np.array(y_train[i]["f_max"])[j] - np.array(y_train[i]["f_min"])[j]],
            # 0默认为背景
            'category_id': int(np.array(y_train[i]["species_id"])[j]) + 1,
            # 为每个object编制唯一id
            'id': bdboxcount,
            'image_id': imgcount,
            'iscrowd': 0,
            # [x1, y1, x2, y1, x2, y2, x1, y2]
            'segmentation': [[np.array(y_train[i]["t_min"])[j],
                              154 - np.array(y_train[i]["f_min"])[j],
                              np.array(y_train[i]["t_max"])[j],
                              154 - np.array(y_train[i]["f_min"])[j],
                              np.array(y_train[i]["t_max"])[j],
                              154 - np.array(y_train[i]["f_max"])[j],
                              np.array(y_train[i]["t_min"])[j],
                              154 - np.array(y_train[i]["f_max"])[j]]]
        })
        bdboxcount += 1
    imgcount += 1
print("COCO dataset image png files copy successfully!")
for i in range(25):
    coco_dataset['categories'].append({'id': i,
                                       'name': i,
                                       'supercategory': i})
filename = "./coco/annotations/instances_train2017.json"
with open(filename, 'w') as file_obj:
    json.dump(coco_dataset, file_obj, indent=1)
print("COCO instances_train2017.json generate successfully!")

coco_dataset = {'categories': [], 'images': [], 'annotations': []}
for i in range(len(X_validate)):
    copyfile(img_dataset + X_validate[i], "./coco/images/val2017/" + X_validate[i])
    coco_dataset['images'].append({'file_name': X_validate[i],
                                   'id': imgcount,
                                   'width': 387,
                                   'height': 154})
    for j in range(y_validate[i].shape[0]):
        coco_dataset['annotations'].append({
            'area': (np.array(y_validate[i]["t_max"])[j] - np.array(y_validate[i]["t_min"])[j]) * (
                    np.array(y_validate[i]["f_max"])[j] - np.array(y_validate[i]["f_min"])[j]),
            # x1, y1为左上角顶点
            # bbox: x1, y1, width, height
            'bbox': [np.array(y_validate[i]["t_min"])[j], 154 - np.array(y_validate[i]["f_max"])[j],
                     np.array(y_validate[i]["t_max"])[j] - np.array(y_validate[i]["t_min"])[j],
                     np.array(y_validate[i]["f_max"])[j] - np.array(y_validate[i]["f_min"])[j]],
            # 0默认为背景
            'category_id': int(np.array(y_validate[i]["species_id"])[j]) + 1,
            # 为每个object编制唯一id
            'id': bdboxcount,
            'image_id': imgcount,
            'iscrowd': 0,
            # [x1, y1, x2, y1, x2, y2, x1, y2]
            'segmentation': [[np.array(y_validate[i]["t_min"])[j],
                              154 - np.array(y_validate[i]["f_min"])[j],
                              np.array(y_validate[i]["t_max"])[j],
                              154 - np.array(y_validate[i]["f_min"])[j],
                              np.array(y_validate[i]["t_max"])[j],
                              154 - np.array(y_validate[i]["f_max"])[j],
                              np.array(y_validate[i]["t_min"])[j],
                              154 - np.array(y_validate[i]["f_max"])[j]]]
        })
        bdboxcount += 1
    imgcount += 1
for i in range(25):
    coco_dataset['categories'].append({'id': i,
                                       'name': i,
                                       'supercategory': i})
filename = "./coco/annotations/instances_val2017.json"
with open(filename, 'w') as file_obj:
    json.dump(coco_dataset, file_obj, indent=1)
print("COCO instances_val2017.json generate successfully!")

coco_dataset = {'categories': [], 'images': [], 'annotations': []}
for i in range(len(X_test)):
    copyfile(img_dataset + X_test[i], "./coco/images/test2017/" + X_test[i])
    coco_dataset['images'].append({'file_name': X_test[i],
                                   'id': imgcount,
                                   'width': 387,
                                   'height': 154})
    for j in range(y_test[i].shape[0]):
        coco_dataset['annotations'].append({
            'area': (np.array(y_test[i]["t_max"])[j] - np.array(y_test[i]["t_min"])[j]) * (
                    np.array(y_test[i]["f_max"])[j] - np.array(y_test[i]["f_min"])[j]),
            # x1, y1为左上角顶点
            # bbox: x1, y1, width, height
            'bbox': [np.array(y_test[i]["t_min"])[j], 154 - np.array(y_test[i]["f_max"])[j],
                     np.array(y_test[i]["t_max"])[j] - np.array(y_test[i]["t_min"])[j],
                     np.array(y_test[i]["f_max"])[j] - np.array(y_test[i]["f_min"])[j]],
            # 0默认为背景
            'category_id': int(np.array(y_test[i]["species_id"])[j]) + 1,
            # 为每个object编制唯一id
            'id': bdboxcount,
            'image_id': imgcount,
            'iscrowd': 0,
            # [x1, y1, x2, y1, x2, y2, x1, y2]
            'segmentation': [[np.array(y_test[i]["t_min"])[j],
                              154 - np.array(y_test[i]["f_min"])[j],
                              np.array(y_test[i]["t_max"])[j],
                              154 - np.array(y_test[i]["f_min"])[j],
                              np.array(y_test[i]["t_max"])[j],
                              154 - np.array(y_test[i]["f_max"])[j],
                              np.array(y_test[i]["t_min"])[j],
                              154 - np.array(y_test[i]["f_max"])[j]]]
        })
        bdboxcount += 1
    imgcount += 1
for i in range(25):
    coco_dataset['categories'].append({'id': i,
                                       'name': i,
                                       'supercategory': i})
filename = "./coco/annotations/instances_test2017.json"
with open(filename, 'w') as file_obj:
    json.dump(coco_dataset, file_obj, indent=1)
print("COCO instances_test2017.json generate successfully!")

# create darknet dataset
darknet_dataset_location = "./darknet/data/kaggle"
file = open("./darknet/data/trainval.txt", 'w')
for i in range(len(X_train_validate)):
    file.write("./data/kaggle/" + X_train_validate[i] + "\n")
file.close()
file = open("./darknet/data/train.txt", 'w')
for i in range(len(X_train)):
    file.write("data/kaggle/" + X_train[i] + "\n")
file.close()
print("Darknet train.txt generate successfully!")
file = open("./darknet/data/val.txt", 'w')
for i in range(len(X_validate)):
    file.write("data/kaggle/" + X_validate[i] + "\n")
file.close()
print("Darknet val.txt generate successfully!")
file = open("./darknet/data/test.txt", 'w')
for i in range(len(X_test)):
    file.write("data/kaggle/" + X_test[i] + "\n")
file.close()
print("Darknet test.txt generate successfully!")
# darknet中的种类从0开始
for i in range(len(Label[0])):
    (filename, extension) = os.path.splitext(Label[0][i])
    file = open("./darknet/data/kaggle/" + filename + ".txt", 'w')
    for j in range(Label[1][i].shape[0]):
        species_id = int(np.array(Label[1][i]["species_id"])[j])
        width = (np.array(Label[1][i]["t_max"])[j] - np.array(Label[1][i]["t_min"])[j]) / 387
        height = (np.array(Label[1][i]["f_max"])[j] - np.array(Label[1][i]["f_min"])[j]) / 154
        center_x = ((np.array(Label[1][i]["t_max"])[j] + np.array(Label[1][i]["t_min"])[j]) / 2) / 387
        center_y = ((308 - np.array(Label[1][i]["f_max"])[j] - np.array(Label[1][i]["f_min"])[j]) / 2) / 154
        file.write(str(species_id) + " " + str(center_x) + " " + str(center_y) +
                   " " + str(width) + " " + str(height))
    file.close()
print("Darknet label txt files generate successfully!")
