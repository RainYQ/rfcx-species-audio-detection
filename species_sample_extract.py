import dataset_transform
import os
from tqdm import tqdm
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2

color_list = ["#FF0000", "#FFFF00", "#008B8B", "#7FFFD4", "#FFFAFA", "#0000FF", "#8A2BE2", "#A52A2A", "#000000",
              "#7FFF00", "#80000040", "#FF7F50", "#6495ED", "#DC143C", "#00FFFF", "#B8860B", "#A9A9A9", "#006400",
              "#FFDAB9", "#8B008B", "#FF00FF", "#483D8B", "#2F4F4F", "#D2B48C"]

Label_train = dataset_transform.voc_reader("./VOC2007_test", "train")
X_train = Label_train[0]
y_train = Label_train[1]
Label_validate = dataset_transform.voc_reader("./VOC2007_test", "val")
X_validate = Label_validate[0]
y_validate = Label_validate[1]
Label_test = dataset_transform.voc_reader("./VOC2007_test", "test")
X_test = Label_test[0]
y_test = Label_test[1]
Label = [[], [], []]
Label[0] = X_train + X_validate + X_test
Label[1] = y_train + y_validate + y_test
imglist = os.listdir("./train_img")
if not os.path.exists("./species_sample/"):
    os.makedirs("./species_sample/")
for i in range(24):
    if not os.path.exists("./species_sample/" + str(i + 1)):
        os.makedirs("./species_sample/" + str(i + 1))
for img in tqdm(imglist):
    test_img = matplotlib.image.imread("./train_img/" + img)
    try:
        data = np.array(Label[1][Label[0].index(img)]
                        [["t_min", "t_max", "f_min", "f_max", "species_id"]])
    except KeyError:
        print(img)
        print(Label[1][Label[0].index(img)])
    # 对于图像坐标轴，以左上角为圆点，垂直向下为y轴，水平向右为x轴，hlines绘制平行于x轴的水平线
    plt.figure()
    plt.imshow(test_img)
    plt.axis('off')
    for i in range(data.shape[0]):
        img_data = cv2.imread("./train_img/" + img)
        x_min = int(round(float(data[i][0])))
        x_max = int(round(float(data[i][1])))
        y_min = int(round(float(154 - data[i][3])))
        y_max = int(round(float(154 - data[i][2])))
        cut = img_data[y_min:y_max, x_min:x_max, :]
        name, suffix = os.path.splitext(img)
        cv2.imwrite("./species_sample/" + str(int(data[i][4]) + 1) + '/' + name + '_' + str(i) + '.png', cut)
    for i in range(data.shape[0]):
        plt.hlines(154 - data[i][2], data[i][0], data[i][1], color_list[int(data[i][4])])
        plt.hlines(154 - data[i][3], data[i][0], data[i][1], color_list[int(data[i][4])])
        plt.vlines(data[i][0], 154 - data[i][2], 154 - data[i][3], color_list[int(data[i][4])])
        plt.vlines(data[i][1], 154 - data[i][2], 154 - data[i][3], color_list[int(data[i][4])])
        plt.text(data[i][0], 154 - data[i][3] - 1.0, str(int(data[i][4]) + 1), color="w")
    plt.savefig("./label_test/" + img, bbox_inches="tight", pad_inches=0.0)
