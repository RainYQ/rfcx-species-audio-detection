import pandas as pd
import numpy as np
import os
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

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

Label = [[], [], []]

for img in tqdm(imglist):
    cur_img = "./train_tran/" + img
    load_img = "./train_img/" + img
    data = label_data.loc[label_data["location"] == cur_img][["t_min", "t_max", "f_min", "f_max", "species_id",
                                                              "songtype_id"]]
    Label[0].append(cv2.imread(load_img) / 255)
    label = []
    for j in range(data.shape[0]):
        label.append(keras.utils.to_categorical([np.uint8(np.array(data["species_id"])[j])], num_classes=24))
    label_sum = np.sum(np.array(label), axis=0).reshape(24)
    for j in range(label_sum.shape[0]):
        if label_sum[j] > 1:
            label_sum[j] = 1
    Label[1].append(label_sum)
    Label[2].append(int(np.array(data["species_id"])[0]))

# 按照6:2:2分配训练集、验证集、测试集
X_train_validate, X_test, y_train_validate, y_test = train_test_split(Label[0], Label[1],
                                                                      stratify=Label[2],
                                                                      test_size=0.2, random_state=SEED)
resnet = keras.applications.EfficientNetB4(weights="imagenet", include_top=False, input_shape=(X_train_validate[0].shape[0],
                                                                                           X_train_validate[0].shape[1],
                                                                                           3), classes=24)
model = keras.models.Sequential()
model.add(resnet)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=24, activation='sigmoid'))
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-6,
                                  amsgrad=True, clipnorm=1.)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
model.load_weights('./model/checkpoints/inception.065-0.13.hdf5')
X_train_validate = np.array(X_train_validate).reshape(-1, X_train_validate[0].shape[0], X_train_validate[0].shape[1], 3)
X_test = np.array(X_test).reshape(-1, X_train_validate[0].shape[0], X_train_validate[0].shape[1], 3)
y_train_validate = np.array(y_train_validate)
y_test = np.array(y_test)
scores = model.evaluate(X_test, y_test, verbose=1, batch_size=8)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
predict = model.predict(X_test, batch_size=8)
print(predict)
