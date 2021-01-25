import pandas as pd
import numpy as np
import os
import tensorflow.keras as keras
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib

SEED = 2021
train_audio_path = "./train"
test_audio_path = "./test"
label_path = "./label"

train_tp = "./train_tp.csv"
train_fp = "./train_fp.csv"
sample_submission = "./sample_submission.csv"

train_data_tp = pd.read_csv(train_tp)
train_data_fp = pd.read_csv(train_fp)

imglist = os.listdir("./train_img")

data = train_data_tp[["recording_id", "species_id"]]
label_data = pd.read_table("./label/label.txt", sep="\t")
file = open("./label/multi-label.txt", 'w')
multi_label = []
for img in tqdm(imglist):
    cur_img = "./train_tran/" + img
    load_img = "./train_img/" + img
    data = label_data.loc[label_data["location"] == cur_img][["t_min", "t_max", "f_min", "f_max", "species_id",
                                                              "songtype_id"]]
    label = []
    for j in range(data.shape[0]):
        label.append(keras.utils.to_categorical([np.uint8(np.array(data["species_id"])[j])], num_classes=24))
    label_sum = np.sum(np.array(label), axis=0).reshape(24)
    for j in range(label_sum.shape[0]):
        if label_sum[j] > 1:
            label_sum[j] = 1
    multi_label.append(label_sum)
    file.write(cur_img + "\t" + str(label_sum) + "\n")
print(np.sum(np.array(multi_label), axis=0))
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.plot(np.linspace(0, 24, 24), np.sum(np.array(multi_label), axis=0).reshape(24))
plt.xlabel("物种")
plt.ylabel("物种标记频数")
plt.title("物种标记频数分布直方图")
plt.axis([0, 24, 0, 120])
plt.show()
