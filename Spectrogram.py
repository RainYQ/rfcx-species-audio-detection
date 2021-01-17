import librosa.core as lc
import numpy as np
import librosa.display
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import math

audio_path = "./train"
train_tp = "./train_tp.csv"
train_fp = "./train_fp.csv"
sample_submission = "./sample_submission.csv"

train_data_tp = pd.read_csv(train_tp)
train_data_fp = pd.read_csv(train_fp)

print("min(f_min) in tp: ", min(train_data_tp["f_min"]), "Hz")
print("max(f_max) in tp: ", max(train_data_tp["f_max"]), "Hz")

print("---------------------------------------------------------")

print("min(f_min) in fp: ", min(train_data_fp["f_min"]), "Hz")
print("max(f_max) in fp: ", max(train_data_fp["f_max"]), "Hz")

print("---------------------------------------------------------")

print("global f_min: ", min([min(train_data_tp["f_min"]), min(train_data_fp["f_min"])]), "Hz")
print("global f_max: ", max([max(train_data_fp["f_max"]), max(train_data_fp["f_max"])]), "Hz")

print("---------------------------------"
      "------------------------")

# 确定频率范围 93.75Hz - 13687.5Hz
sample_id = "0099c367b"
sample_flac_path = "./train/0099c367b.flac"
print("data in train_tp:")
if len(train_data_tp.loc[train_data_tp["recording_id"] == "0099c367b"]) != 0:
    t_min = np.array(train_data_tp.loc[train_data_tp["recording_id"] == "0099c367b"]["t_min"])
    t_max = np.array(train_data_tp.loc[train_data_tp["recording_id"] == "0099c367b"]["t_max"])
    f_min = np.array(train_data_tp.loc[train_data_tp["recording_id"] == "0099c367b"]["f_min"])
    f_max = np.array(train_data_tp.loc[train_data_tp["recording_id"] == "0099c367b"]["f_max"])
    print(t_min)
    print(t_max)
    print(f_min)
    print(f_max)
else:
    print("NULL")
print("data in train_fp:")
if len(train_data_fp.loc[train_data_fp["recording_id"] == "0099c367b"]) != 0:
    t_min = np.array(train_data_fp.loc[train_data_fp["recording_id"] == "0099c367b"]["t_min"])
    t_max = np.array(train_data_fp.loc[train_data_fp["recording_id"] == "0099c367b"]["t_max"])
    f_min = np.array(train_data_fp.loc[train_data_fp["recording_id"] == "0099c367b"]["f_min"])
    f_max = np.array(train_data_fp.loc[train_data_fp["recording_id"] == "0099c367b"]["f_max"])
    print(t_min)
    print(t_max)
    print(f_min)
    print(f_max)
else:
    print("NULL")
print("---------------------------------------------------------")
y_, fs = sf.read(sample_flac_path)
print("flac file sampling frequency: ", fs, "Hz")
print("---------------------------------------------------------")
n_fft = 2048
# 限制最大频率为2**14 = 16384Hz
# sr=None表示以原始分辨率进行采样
y, sr = librosa.load(sample_flac_path, sr=None)

y = y[math.floor((t_min[0] - 2) * fs):math.ceil((t_max[0] + 2) * fs)]

# 获取窄带声谱图
# 实验证明应该使用窄带声谱图
spect = librosa.feature.melspectrogram(y=y, sr=fs, n_fft=2048, win_length=512, hop_length=128)
mel_spect = librosa.power_to_db(spect, ref=np.max)
plt.figure(figsize=(5, 2))
librosa.display.specshow(mel_spect, y_axis='mel', sr=fs, fmin=0, fmax=16384, x_axis='s', hop_length=128);
plt.hlines(f_min[0], 2, 2 + t_max[0] - t_min[0])
plt.hlines(f_max[0], 2, 2 + t_max[0] - t_min[0])
plt.vlines(2, f_min[0], f_max[0])
plt.vlines(2 + t_max[0] - t_min[0], f_min[0], f_max[0])
plt.axis('off')
plt.savefig("narrowband spectrogram.png", bbox_inches="tight", pad_inches=0.0)
# plt.colorbar(format='%+2.0f dB')
# plt.title('narrowband spectrogram')
plt.show()