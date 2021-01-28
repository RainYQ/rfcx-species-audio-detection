from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import os
from lxml.etree import Element, SubElement, ElementTree
import json
import shutil
from shutil import copyfile

try:
    import xml.etree.cElementTree as xmlElementTree
except ImportError:
    import xml.etree.ElementTree as xmlElementTree

train_tp = "./train_tp.csv"
train_fp = "./train_fp.csv"
train_data_tp = pd.read_csv(train_tp)
train_data_fp = pd.read_csv(train_fp)
label_data = pd.read_table("./label/label.txt", sep="\t")
img_dataset = "./train_img/"


def coco_reader(json_path):
    Label = [[], [], []]
    json_data = json.load(open(json_path, 'r'))
    for i in range(len(json_data['images'])):
        image_id = json_data['images'][i]['id']
        coco = COCO(json_path)
        image_data = coco.loadImgs([image_id])
        file_name = image_data[0]['file_name']
        Label[0].append(file_name)
        annotation = []
        for ann in json_data['annotations']:
            if ann['image_id'] == image_id:
                # x1, y1, width, height
                # t_min, 154 - f_max, t_max - t_min, f_max - f_min
                # 转换到COCO数据集格式的时候已经丢失songtype_id信息
                annotation.append({'t_min': ann['bbox'][0], 't_max': ann['bbox'][0] + ann['bbox'][2],
                                   'f_min': 154 - ann['bbox'][1] - ann['bbox'][3], 'f_max': 154 - ann['bbox'][1],
                                   'species_id': ann['category_id'] - 1})
        Label[1].append(pd.DataFrame(annotation))
    return Label


def coco_writer(X_train, y_train, X_validate, y_validate, X_test, y_test, coco_path):
    coco_dataset = {'categories': [], 'images': [], 'annotations': []}
    imgcount = 0
    bdboxcount = 0
    if not os.path.exists(coco_path):
        os.makedirs(coco_path)
    if not os.path.exists(coco_path + '/annotations/'):
        os.makedirs(coco_path + '/annotations/')
    if not os.path.exists(coco_path + '/images/'):
        os.makedirs(coco_path + '/images/')
    if not os.path.exists(coco_path + '/images/train2017/'):
        os.makedirs(coco_path + '/images/train2017/')
    if not os.path.exists(coco_path + '/images/val2017/'):
        os.makedirs(coco_path + '/images/val2017/')
    if not os.path.exists(coco_path + '/images/test2017/'):
        os.makedirs(coco_path + '/images/test2017/')
    if len(os.listdir(coco_path + "/images/train2017/")):
        shutil.rmtree(coco_path + "/images/train2017/")
        os.mkdir(coco_path + "/images/train2017/")
    if len(os.listdir(coco_path + "/images/val2017/")):
        shutil.rmtree(coco_path + "/images/val2017/")
        os.mkdir(coco_path + "/images/val2017/")
    if len(os.listdir(coco_path + "/images/test2017/")):
        shutil.rmtree(coco_path + "/images/test2017/")
        os.mkdir(coco_path + "/images/test2017/")
    coco_train_val_dataset = {'categories': [], 'images': [], 'annotations': []}
    for i in range(len(X_train)):
        copyfile(img_dataset + X_train[i], coco_path + '/images/train2017/' + X_train[i])
        coco_dataset['images'].append({'file_name': X_train[i],
                                       'id': imgcount,
                                       'width': 387,
                                       'height': 154})
        coco_train_val_dataset['images'].append({'file_name': X_train[i],
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
            coco_train_val_dataset['annotations'].append({
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
    filename = coco_path + "/annotations/instances_train2017.json"
    with open(filename, 'w') as file_obj:
        json.dump(coco_dataset, file_obj, indent=1)
    print("COCO instances_train2017.json generate successfully!")
    coco_dataset = {'categories': [], 'images': [], 'annotations': []}
    for i in range(len(X_validate)):
        copyfile(img_dataset + X_validate[i], coco_path + "/images/val2017/" + X_validate[i])
        coco_dataset['images'].append({'file_name': X_validate[i],
                                       'id': imgcount,
                                       'width': 387,
                                       'height': 154})
        coco_train_val_dataset['images'].append({'file_name': X_validate[i],
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
            coco_train_val_dataset['annotations'].append({
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
    filename = coco_path + "/annotations/instances_val2017.json"
    with open(filename, 'w') as file_obj:
        json.dump(coco_dataset, file_obj, indent=1)
    print("COCO instances_val2017.json generate successfully!")
    for i in range(25):
        coco_train_val_dataset['categories'].append({'id': i,
                                                     'name': i,
                                                     'supercategory': i})
    filename = coco_path + "/annotations/instances_trainval2017.json"
    with open(filename, 'w') as file_obj:
        json.dump(coco_train_val_dataset, file_obj, indent=1)
    print("COCO instances_trainval2017.json generate successfully!")

    coco_dataset = {'categories': [], 'images': [], 'annotations': []}
    for i in range(len(X_test)):
        copyfile(img_dataset + X_test[i], coco_path + "/images/test2017/" + X_test[i])
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
    filename = coco_path + "/annotations/instances_test2017.json"
    with open(filename, 'w') as file_obj:
        json.dump(coco_dataset, file_obj, indent=1)
    print("COCO instances_test2017.json generate successfully!")


def voc_reader(voc_path, mode):
    Label = [[], [], []]
    train_data = open(voc_path + "/ImageSets/Main/" + mode + ".txt")
    line = train_data.readline()
    while line:
        f = line[:-1].split("\t")[0]  # 除去末尾的换行符
        f += ".xml"
        tree = xmlElementTree.parse(voc_path + "/Annotations/" + f)  # 打开xml文档
        root = tree.getroot()  # 获得root节点
        filename = root.find('filename').text
        (filename, extension) = os.path.splitext(filename)
        filename = filename + ".png"
        Label[0].append(filename)
        annotation = []
        for object in root.findall('object'):  # 找到root节点下的所有object节点
            name = object.find('name').text  # 子节点下节点name的值
            try:
                species_id = int(name) - 1
            except ValueError:
                print(filename + " species_id error!")
                continue
            bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
            difficult = object.find('difficult').text
            xmin = bndbox.find('xmin').text
            t_min = float(xmin)
            ymin = bndbox.find('ymin').text
            f_max = 154 - float(ymin)
            xmax = bndbox.find('xmax').text
            t_max = float(xmax)
            ymax = bndbox.find('ymax').text
            f_min = 154 - float(ymax)
            annotation.append({'t_min': t_min, 't_max': t_max, 'f_min': f_min, 'f_max': f_max,
                               'species_id': species_id, 'difficult': difficult})
        Label[1].append(pd.DataFrame(annotation))
        line = train_data.readline()
    return Label


def voc_writer(X_train, y_train, X_validate, y_validate, X_test, y_test, voc_path):
    xml_location = voc_path + "/Annotations/"
    # 制作VOC格式的label
    # warning: VOC格式中的种类从1开始
    if not os.path.exists(voc_path):
        os.makedirs(voc_path)
    if not os.path.exists(voc_path + '/Annotations/'):
        os.makedirs(voc_path + '/Annotations/')
    if not os.path.exists(voc_path + '/ImageSets/'):
        os.makedirs(voc_path + '/ImageSets/')
    if not os.path.exists(voc_path + '/ImageSets/Main/'):
        os.makedirs(voc_path + '/ImageSets/Main/')
    if not os.path.exists(voc_path + '/JPEGImages/'):
        os.makedirs(voc_path + '/JPEGImages/')
    for k in [[X_train, y_train], [X_validate, y_validate], [X_test, y_test]]:
        for i in range(len(k[0])):
            copyfile(img_dataset + k[0][i], voc_path + '/JPEGImages/' + k[0][i])
            node_root = Element('annotation')
            node_folder = SubElement(node_root, 'folder')
            node_folder.text = os.path.basename(voc_path)
            node_filename = SubElement(node_root, 'filename')
            node_filename.text = k[0][i]
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
            for j in range(k[1][i].shape[0]):
                # xml file generator
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                # 0默认为背景
                node_name.text = str(int(np.array(k[1][i]["species_id"])[j]) + 1)
                node_difficult = SubElement(node_object, 'difficult')
                if 'difficult' in k[1][i].columns:
                    node_difficult.text = str(int(np.array(k[1][i]["difficult"])[j]))
                else:
                    node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                # 在VOC2007 Dataset中, 坐标系与图像坐标系相同，以左上角为圆点，垂直向下为y轴，水平向右为x轴
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(np.array(k[1][i]["t_min"])[j])
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(154 - np.array(k[1][i]["f_max"])[j])
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text = str(np.array(k[1][i]["t_max"])[j])
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(154 - np.array(k[1][i]["f_min"])[j])
            doc = ElementTree(node_root)
            (filename, extension) = os.path.splitext(k[0][i])
            # xml file writer
            # pretty_print 正确换行
            # xml_declaration False 去除xml文件声明
            doc.write(open(xml_location + filename + ".xml", "wb"), encoding="utf-8", xml_declaration=False,
                      pretty_print=True)
    print("VOC Annotations xml files generate successfully!")
    # 制作VOC格式的train.txt val.txt trainval.txt test.txt
    # VOC格式要求第一列为文件名（去除后缀名）, 第二列为±1, 1表示正样本, -1表示负样本
    file = open(voc_path + '/ImageSets/Main/trainval.txt', 'w')
    for i in range(len(X_train)):
        (filename, extension) = os.path.splitext(X_train[i])
        file.write(filename + "\t" + "1" + "\n")
    for i in range(len(X_validate)):
        (filename, extension) = os.path.splitext(X_validate[i])
        file.write(filename + "\t" + "1" + "\n")
    file.close()
    print("VOC trainval.txt generate successfully!")
    file = open(voc_path + '/ImageSets/Main/train.txt', 'w')
    for i in range(len(X_train)):
        (filename, extension) = os.path.splitext(X_train[i])
        file.write(filename + "\t" + "1" + "\n")
    file.close()
    print("VOC train.txt generate successfully!")
    file = open(voc_path + '/ImageSets/Main/val.txt', 'w')
    for i in range(len(X_validate)):
        (filename, extension) = os.path.splitext(X_validate[i])
        file.write(filename + "\t" + "1" + "\n")
    file.close()
    print("VOC val.txt generate successfully!")
    file = open(voc_path + '/ImageSets/Main/test.txt', 'w')
    for i in range(len(X_test)):
        (filename, extension) = os.path.splitext(X_test[i])
        file.write(filename + "\t" + "1" + "\n")
    file.close()
    print("VOC test.txt generate successfully!")


def darknet_reader(darknet_path, mode):
    Label = [[], [], []]
    k = pd.read_table(darknet_path + "/data/" + mode + ".txt", header=None).values.tolist()
    for j in range(len(k)):
        file_name = os.path.basename(k[j][0])
        Label[0].append(file_name)
        (filename, extension) = os.path.splitext(file_name)
        filename += ".txt"
        data = pd.read_table(darknet_path + "/data/kaggle/" + filename, sep=" ", header=None)
        annotation = []
        for z in range(data.shape[0]):
            # species_id width height center_x center_y
            # species_id (t_max - t_min) / 387 (f_max - f_min) / 154 ((t_max + t_min) / 2) / 387 ((308 - f_max - f_min) / 2) / 154
            # species_id t_min t_max f_min f_max
            # species_id (center_x - width / 2) * 387 (center_x + width / 2) * 154 154 - (center_y + height / 2) * 154 154 - (center_y - height / 2) * 154
            width = float(data.values[z][3])
            height = float(data.values[z][4])
            center_x = float(data.values[z][1])
            center_y = float(data.values[z][2])
            annotation.append({'t_min': (center_x - width / 2) * 387, 't_max': (center_x + width / 2) * 387,
                               'f_min': 154 - (center_y + height / 2) * 154,
                               'f_max': 154 - (center_y - height / 2) * 154,
                               'species_id': int(data[z][0])})
        Label[1].append(pd.DataFrame(annotation))
    return Label


def darknet_writer(X_train, y_train, X_validate, y_validate, X_test, y_test, darknet_path):
    if not os.path.exists(darknet_path):
        os.makedirs(darknet_path)
    if not os.path.exists(darknet_path + '/data/'):
        os.makedirs(darknet_path + '/data/')
    if not os.path.exists(darknet_path + '/data/kaggle/'):
        os.makedirs(darknet_path + '/data/kaggle/')
    # create darknet dataset
    file = open(darknet_path + "/data/trainval.txt", 'w')
    for i in range(len(X_train)):
        file.write("data/kaggle/" + X_train[i] + "\n")
    for i in range(len(X_validate)):
        file.write("data/kaggle/" + X_validate[i] + "\n")
    file.close()
    file = open(darknet_path + "/data/train.txt", 'w')
    for i in range(len(X_train)):
        file.write("data/kaggle/" + X_train[i] + "\n")
    file.close()
    print("Darknet train.txt generate successfully!")
    file = open(darknet_path + "/data/val.txt", 'w')
    for i in range(len(X_validate)):
        file.write("data/kaggle/" + X_validate[i] + "\n")
    file.close()
    print("Darknet val.txt generate successfully!")
    file = open(darknet_path + "/data/test.txt", 'w')
    for i in range(len(X_test)):
        file.write("data/kaggle/" + X_test[i] + "\n")
    file.close()
    print("Darknet test.txt generate successfully!")
    # darknet中的种类从0开始
    for k in [[X_train, y_train], [X_validate, y_validate], [X_test, y_test]]:
        for i in range(len(k[0])):
            copyfile(img_dataset + k[0][i], darknet_path + '/data/kaggle/' + k[0][i])
            (filename, extension) = os.path.splitext(k[0][i])
            file = open(darknet_path + "/data/kaggle/" + filename + ".txt", 'w')
            for j in range(k[1][i].shape[0]):
                species_id = int(np.array(k[1][i]["species_id"])[j])
                width = (np.array(k[1][i]["t_max"])[j] - np.array(k[1][i]["t_min"])[j]) / 387
                height = (np.array(k[1][i]["f_max"])[j] - np.array(k[1][i]["f_min"])[j]) / 154
                center_x = ((np.array(k[1][i]["t_max"])[j] + np.array(k[1][i]["t_min"])[j]) / 2) / 387
                center_y = ((308 - np.array(k[1][i]["f_max"])[j] - np.array(k[1][i]["f_min"])[j]) / 2) / 154
                file.write(str(species_id) + " " + str(center_x) + " " + str(center_y) +
                           " " + str(width) + " " + str(height) + "\n")
            file.close()
    print("Darknet label txt files generate successfully!")


if __name__ == '__main__':
    # VOC reader sample

    Label_train = voc_reader("./VOC2007", "train")
    X_train = Label_train[0]
    y_train = Label_train[1]
    Label_validate = voc_reader("./VOC2007", "val")
    X_validate = Label_validate[0]
    y_validate = Label_validate[1]
    Label_test = voc_reader("./VOC2007", "test")
    X_test = Label_test[0]
    y_test = Label_test[1]

    # coco reader sample

    # Label_train = coco_reader("./coco/annotations/instances_train2017.json")
    # X_train = Label_train[0]
    # y_train = Label_train[1]
    # Label_validate = coco_reader("./coco/annotations/instances_val2017.json")
    # X_validate = Label_validate[0]
    # y_validate = Label_validate[1]
    # Label_test = coco_reader("./coco/annotations/instances_test2017.json")
    # X_test = Label_test[0]
    # y_test = Label_test[1]

    # coco writer sample

    coco_writer(X_train, y_train, X_validate, y_validate, X_test, y_test, './coco_test')

    # VOC writer sample

    voc_writer(X_train, y_train, X_validate, y_validate, X_test, y_test, "./VOC2007_test")

    # darknet reader sample

    # Label_train = darknet_reader('./darknet', "train")
    # X_train = Label_train[0]
    # y_train = Label_train[1]
    # Label_validate = darknet_reader('./darknet', "val")
    # X_validate = Label_validate[0]
    # y_validate = Label_validate[1]
    # Label_test = darknet_reader('./darknet', "test")
    # X_test = Label_test[0]
    # y_test = Label_test[1]

    # darknet writer sample

    darknet_writer(X_train, y_train, X_validate, y_validate, X_test, y_test, "./darknet_test")
