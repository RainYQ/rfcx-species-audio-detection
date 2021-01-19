import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from lxml.etree import Element, SubElement, ElementTree
import json

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
    data = label_data.loc[label_data["location"] == cur_img]
    [["t_min", "t_max", "f_min", "f_max", "species_id", "songtype_id"]]
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

xml_location = "./VOC2007/VOCdevkit2007/VOC2007/Annotations/"
# 制作VOC格式的label
# 制作COCO格式的json
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
        node_name.text = str(int(np.array(Label[1][i]["species_id"])[j]))
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
file = open('./VOC2007/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt', 'w')
for i in range(len(X_train_validate)):
    (filename, extension) = os.path.splitext(X_train_validate[i])
    file.write(filename + "\t" + "1" + "\n")
file.close()
print("trainval.txt generate successfully!")
file = open('./VOC2007/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt', 'w')
for i in range(len(X_train)):
    (filename, extension) = os.path.splitext(X_train[i])
    file.write(filename + "\t" + "1" + "\n")
file.close()
print("train.txt generate successfully!")
file = open('./VOC2007/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt', 'w')
for i in range(len(X_validate)):
    (filename, extension) = os.path.splitext(X_validate[i])
    file.write(filename + "\t" + "1" + "\n")
file.close()
print("val.txt generate successfully!")
file = open('./VOC2007/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt', 'w')
for i in range(len(X_test)):
    (filename, extension) = os.path.splitext(X_test[i])
    file.write(filename + "\t" + "1" + "\n")
file.close()
print("test.txt generate successfully!")
