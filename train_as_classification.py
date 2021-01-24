import pickle
import pandas as pd
import numpy as np
import os
import tensorflow.keras as keras
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tqdm import tqdm
import scipy.misc
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
checkpointer = ModelCheckpoint(
    filepath=os.path.join('./model/', 'checkpoints',
                          'inception.{epoch:03d}-{val_loss:.2f}.hdf5'),
    verbose=1, monitor='val_loss',
    save_best_only=True)

# Stop when we stop learning.
early_stopper = EarlyStopping(monitor='val_loss', patience=50)

# TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'), write_graph=True, write_grads=False, write_images=True,
                          embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                          embeddings_data=None, update_freq='epoch')

Callback = [checkpointer, early_stopper, tensorboard]

weights = class_weight.compute_class_weight('balanced', np.unique(np.array(train_data_tp["species_id"])),
                                            np.array(train_data_tp["species_id"]))
class_weight_dict = dict(enumerate(weights))

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
# model.build(input_shape=(16, 224, 112, 3))
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
history_save = open('./data/result/Train_History_Dict.txt', 'wb')
X_train_validate = np.array(X_train_validate).reshape(-1, X_train_validate[0].shape[0], X_train_validate[0].shape[1], 3)
X_test = np.array(X_test).reshape(-1, X_train_validate[0].shape[0], X_train_validate[0].shape[1], 3)
y_train_validate = np.array(y_train_validate)
y_test = np.array(y_test)
history = model.fit(X_train_validate, y_train_validate, epochs=2000,
                    batch_size=8, validation_split=0.2, shuffle=True, callbacks=Callback,
                    class_weight=class_weight_dict)
pickle.dump(history.history, history_save)
scores = model.evaluate(X_test, y_test, verbose=1, batch_size=8)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
model.save('./model/model_final.hdf5')
history_save.close()
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper right')
plt.show()
