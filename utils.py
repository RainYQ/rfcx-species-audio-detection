import dataset_transform
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib
from matplotlib import pyplot as plt


# matplotlib.use('TkAgg')
def species_id_distribution():
    Label_train = dataset_transform.voc_reader("./VOC2007_test", "train")
    X_train = Label_train[0]
    y_train = Label_train[1]
    Label_validate = dataset_transform.voc_reader("./VOC2007_test", "val")
    X_validate = Label_validate[0]
    y_validate = Label_validate[1]
    Label_test = dataset_transform.voc_reader("./VOC2007_test", "test")
    X_test = Label_test[0]
    y_test = Label_test[1]

    imglist = os.listdir("./train_img")
    species_id_set_train = []
    species_id_set_validate = []
    species_id_set_test = []
    for img in tqdm(imglist):
        try:
            data = np.array(y_train[X_train.index(img)]["species_id"])
            mode = 0
        except ValueError:
            try:
                data = np.array(y_validate[X_validate.index(img)]["species_id"])
                mode = 1
            except ValueError:
                data = np.array(y_test[X_test.index(img)]["species_id"])
                mode = 2
        label = []
        for j in range(data.shape[0]):
            label.append(keras.utils.to_categorical([np.uint8(data[j])], num_classes=24))
        label_sum = np.sum(np.array(label), axis=0).reshape(24)
        if mode == 0:
            species_id_set_train.append(label_sum)
        elif mode == 1:
            species_id_set_validate.append(label_sum)
        elif mode == 2:
            species_id_set_test.append(label_sum)
    train = np.sum(np.array(species_id_set_train), axis=0).reshape(24)
    print(train)
    validate = np.sum(np.array(species_id_set_validate), axis=0).reshape(24)
    print(validate)
    test = np.sum(np.array(species_id_set_test), axis=0).reshape(24)
    print(test)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    name = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"
            , "16", "17", "18", "19", "20", "21", "22", "23", "24")
    plt.figure()
    plt.bar(np.linspace(0, 24, 24), height=train + validate + test, width=0.5, label="物种标记频数", tick_label=name,
            color='g')
    plt.bar(np.linspace(0, 24, 24), height=train + validate, width=0.5, label="物种标记频数", tick_label=name, color='b')
    plt.bar(np.linspace(0, 24, 24), height=train, width=0.5, label="物种标记频数", tick_label=name, color='r')
    plt.xlabel("物种")
    plt.ylabel("物种标记频数")
    plt.title("物种标记频数分布直方图")
    plt.show()


species_id_distribution()
