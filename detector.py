import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tensorflow.keras as keras
import PIL.Image as Image
import cv2
import csv
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
from io import BytesIO
import PIL


def model():
    resnet = keras.applications.EfficientNetB4(weights="imagenet", include_top=False,
                                               input_shape=(154, 387, 3), classes=24)
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
    model.load_weights('./model/checkpoints/inception.065-0.13.hdf5')
    # model.summary()
    return model


test_audio_path = "./test/"
test_audio_list = os.listdir(test_audio_path)
det_model = model()
result_csv = open('result.csv','w',encoding='utf-8')
csv_writer = csv.writer(result_csv)
csv_writer.writerow(["recording_id", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12",
                     "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23"])
for sample in tqdm(test_audio_list):
    (recording_id, extension) = os.path.splitext(sample)
    y, sr = librosa.load(test_audio_path + sample, sr=None)
    result = []
    for i in range(6):
        cut = y[sr * 10 * i:sr * 10 * (i + 1)]
        spect = librosa.feature.melspectrogram(y=cut, sr=sr, n_fft=2048, win_length=512, hop_length=128)
        mel_spect = librosa.power_to_db(spect, ref=np.max)
        f = plt.figure(figsize=(5, 2))
        plt.axis('off')
        librosa.display.specshow(mel_spect, y_axis='mel', sr=sr, fmin=0, fmax=16384, x_axis='s', hop_length=128)
        # using disk read/write io
        # plt.savefig("./tmp.png", bbox_inches="tight", pad_inches=0.0)
        # data = cv2.imread("./tmp.png")
        # using RAM read/write io
        # 申请缓冲地址
        buffer_ = BytesIO()  # using buffer,great way!
        plt.savefig(buffer_, bbox_inches="tight", pad_inches=0.0)
        buffer_.seek(0)
        # 用PIL从内存中读取
        dataPIL = PIL.Image.open(buffer_)
        data = cv2.cvtColor(np.asarray(dataPIL), cv2.COLOR_RGB2BGR)
        data = np.array(data).reshape([1, 154, 387, 3])
        # 释放缓存
        probability = det_model.predict(data)
        buffer_.close()
        f.clear()
        plt.close(f)
        result.append(probability)
    r = np.max(np.array(result), axis=0).reshape(24)
    data_list = [recording_id]
    for j in range(24):
        data_list.append(str(format(r[j], '.4f')))
    csv_writer.writerow(data_list)
result_csv.close()

