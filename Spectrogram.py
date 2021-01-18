import librosa.core as lc
import numpy as np
import librosa.display
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import random
from tqdm import tqdm
import matplotlib

# 24个物种对应24种颜色
color_list = ["#FF0000", "#FFFF00", "#008B8B", "#7FFFD4", "#FFFAFA", "#0000FF", "#8A2BE2", "#A52A2A", "#000000",
              "#7FFF00", "#80000040", "#FF7F50", "#6495ED", "#DC143C", "#00FFFF", "#B8860B", "#A9A9A9", "#006400",
              "#FFDAB9", "#8B008B", "#FF00FF", "#483D8B", "#2F4F4F", "#D2B48C"]
train_audio_path = "./train"
test_audio_path = "./test"
label_path = "./label"
train_tp = "./train_tp.csv"
train_fp = "./train_fp.csv"
sample_submission = "./sample_submission.csv"
train_data_tp = pd.read_csv(train_tp)
train_data_fp = pd.read_csv(train_fp)
n_fft = 2048
sample_id = "d59d099b3"
sample_flac_path = os.path.join(train_audio_path, sample_id + ".flac")
y_, fs = sf.read(sample_flac_path)
print("---------------------------------------------------------")
print("flac file sampling frequency: ", fs, "Hz")
print("---------------------------------------------------------")


def test():
    # mode = "show"
    mode = "save only"
    # clip_mode = "none"
    clip_mode = "random_clip_10s"
    print("---------------------------------------------------------")
    print("global f_min: ", min([min(train_data_tp["f_min"]), min(train_data_fp["f_min"])]), "Hz")
    print("global f_max: ", max([max(train_data_fp["f_max"]), max(train_data_fp["f_max"])]), "Hz")
    print("---------------------------------------------------------")
    # 确定频率范围 93.75Hz - 13687.5Hz
    print("data in train_tp:")
    if len(train_data_tp.loc[train_data_tp["recording_id"] == sample_id]) != 0:
        data_tp = np.array(train_data_tp.loc[train_data_tp["recording_id"] == sample_id]
                           [["t_min", "t_max", "f_min", "f_max", "species_id", "songtype_id"]])
        print(data_tp)
    else:
        print("NULL")
    print("data in train_fp:")
    if len(train_data_fp.loc[train_data_fp["recording_id"] == sample_id]) != 0:
        data_fp = np.array(train_data_tp.loc[train_data_fp["recording_id"] == sample_id]
                           [["t_min", "t_max", "f_min", "f_max", "species_id", "songtype_id"]])
        print(data_fp)
    else:
        print("NULL")
    print("---------------------------------------------------------")
    # 限制最大频率为2**14 = 16384Hz
    # sr=None表示以原始分辨率进行采样
    y, sr = librosa.load(sample_flac_path, sr=None)
    if clip_mode == "random_clip_10s":
        for i in range(data_tp.shape[0]):
            delta_t = data_tp[i][1] - data_tp[i][0]
            right_delta = y.shape[0] - math.ceil(data_tp[i][1] * fs)
            limit = math.ceil((10 - delta_t) * fs)
            left_delta = math.ceil(data_tp[i][0] * fs)
            min_flag = right_delta < left_delta
            random_data = random.randint(0, min(min(right_delta, left_delta), limit))
            # 从可裁剪范围小的一侧选择随机数，确保可以正确裁剪
            if min_flag:
                y_clip = y[math.floor(data_tp[i][0] * fs - limit + random_data):math.ceil(
                    data_tp[i][1] * fs + random_data)]
                # left_label表示在裁切后，声音标记左边界位置
                left_label = math.floor(data_tp[i][0] * fs) - math.floor(data_tp[i][0] * fs - limit + random_data)
                # right_label表示在裁切后，声音标记右边界位置
                right_label = left_label + math.ceil(delta_t * fs)
            else:
                y_clip = y[math.floor(data_tp[i][0] * fs - random_data):math.ceil(
                    data_tp[i][1] * fs + limit - random_data)]
                left_label = math.floor(data_tp[i][0] * fs) - math.floor(data_tp[i][0] * fs - random_data)
                right_label = left_label + math.ceil(delta_t * fs)
            # 获取窄带声谱图
            # 实验证明应该使用窄带声谱图
            spect = librosa.feature.melspectrogram(y=y_clip, sr=fs, n_fft=2048, win_length=512, hop_length=128)
            mel_spect = librosa.power_to_db(spect, ref=np.max)
            plt.figure(figsize=(5, 2))
            librosa.display.specshow(mel_spect, y_axis='mel', sr=fs, fmin=0, fmax=16384, x_axis='s', hop_length=128)
            # 画框框
            plt.hlines(data_tp[i][2], left_label / fs, right_label / fs, color_list[int(data_tp[i][4])])
            plt.hlines(data_tp[i][3], left_label / fs, right_label / fs, color_list[int(data_tp[i][4])])
            plt.vlines(left_label / fs, data_tp[i][2], data_tp[i][3], color_list[int(data_tp[i][4])])
            plt.vlines(right_label / fs, data_tp[i][2], data_tp[i][3], color_list[int(data_tp[i][4])])
            if mode == "save only":
                plt.axis('off')
                plt.savefig("temp/" + sample_id + str(i) + ".png", bbox_inches="tight", pad_inches=0.0)
            if mode == "show":
                plt.colorbar(format='%+2.0f dB')
                plt.title('narrowband spectrogram')
                plt.show()
    elif clip_mode == "none":
        spect = librosa.feature.melspectrogram(y=y, sr=fs, n_fft=2048, win_length=512, hop_length=128)
        mel_spect = librosa.power_to_db(spect, ref=np.max)
        plt.figure(figsize=(5, 2))
        librosa.display.specshow(mel_spect, y_axis='mel', sr=fs, fmin=0, fmax=16384, x_axis='s', hop_length=128)
        for i in range(data_tp.shape[0]):
            # 画框框
            plt.hlines(data_tp[i][2], data_tp[i][0], data_tp[i][1], color_list[int(data_tp[i][4])])
            plt.hlines(data_tp[i][3], data_tp[i][0], data_tp[i][1], color_list[int(data_tp[i][4])])
            plt.vlines(data_tp[i][0], data_tp[i][2], data_tp[i][3], color_list[int(data_tp[i][4])])
            plt.vlines(data_tp[i][1], data_tp[i][2], data_tp[i][3], color_list[int(data_tp[i][4])])
        if mode == "save only":
            plt.axis('off')
            plt.savefig("temp/narrowband_spectrogram_none_clip.png", bbox_inches="tight", pad_inches=0.0)
        if mode == "show":
            plt.colorbar(format='%+2.0f dB')
            plt.title('narrowband spectrogram')
            plt.show()


def label_verify():
    imglist = os.listdir("./train_tran")
    for img in tqdm(imglist):
        test_img = matplotlib.image.imread("./train_tran/" + img)
        label_data = pd.read_table("./label/label.txt", sep="\t")
        data = np.array(label_data.loc[label_data["location"] == "./train_tran/" + img]
                        [["t_min", "t_max", "f_min", "f_max", "species_id", "songtype_id"]])
        plt.figure()
        plt.imshow(test_img)
        plt.axis('off')
        # 对于图像坐标轴，以左上角为圆点，垂直向下为y轴，水平向右为x轴，hlines绘制平行于x轴的水平线
        plt.hlines(154 - data[0][2], data[0][0], data[0][1], color_list[int(data[0][4])])
        plt.hlines(154 - data[0][3], data[0][0], data[0][1], color_list[int(data[0][4])])
        plt.vlines(data[0][0], 154 - data[0][2], 154 - data[0][3], color_list[int(data[0][4])])
        plt.vlines(data[0][1], 154 - data[0][2], 154 - data[0][3], color_list[int(data[0][4])])
        plt.savefig("./label_test/" + img, bbox_inches="tight", pad_inches=0.0)


def main():
    file = open('./label/label.txt', 'w')
    file.write("location\tt_min\tt_max\tf_min\tf_max\tspecies_id\tsongtype_id\n")
    recording_ids = list(set(train_data_tp["recording_id"]))
    for recording_id in tqdm(recording_ids):
        sample_flac_path = os.path.join(train_audio_path, recording_id + ".flac")
        data_tp = np.array(train_data_tp.loc[train_data_tp["recording_id"] == recording_id]
                           [["t_min", "t_max", "f_min", "f_max", "species_id", "songtype_id"]])
        y, sr = librosa.load(sample_flac_path, sr=None)
        for i in range(data_tp.shape[0]):
            delta_t = data_tp[i][1] - data_tp[i][0]
            right_delta = y.shape[0] - math.ceil(data_tp[i][1] * fs)
            limit = math.ceil((10 - delta_t) * fs)
            left_delta = math.ceil(data_tp[i][0] * fs)
            min_flag = right_delta < left_delta
            random_data = random.randint(0, min(min(right_delta, left_delta), limit))
            if min_flag:
                y_clip = y[math.floor(data_tp[i][0] * fs - limit + random_data):math.ceil(
                    data_tp[i][1] * fs + random_data)]
                left_label = math.floor(data_tp[i][0] * fs) - math.floor(data_tp[i][0] * fs - limit + random_data)
                right_label = left_label + math.ceil(delta_t * fs)
            else:
                y_clip = y[math.floor(data_tp[i][0] * fs - random_data):math.ceil(
                    data_tp[i][1] * fs + limit - random_data)]
                left_label = math.floor(data_tp[i][0] * fs) - math.floor(data_tp[i][0] * fs - random_data)
                right_label = left_label + math.ceil(delta_t * fs)
            spect = librosa.feature.melspectrogram(y=y_clip, sr=fs, n_fft=2048, win_length=512, hop_length=128)
            mel_spect = librosa.power_to_db(spect, ref=np.max)
            plt.figure(figsize=(5, 2))
            librosa.display.specshow(mel_spect, y_axis='mel', sr=fs, fmin=0, fmax=16384, x_axis='s', hop_length=128)
            # 绘制主要label
            plt.hlines(data_tp[i][2], left_label / fs, right_label / fs, color_list[int(data_tp[i][4])])
            plt.hlines(data_tp[i][3], left_label / fs, right_label / fs, color_list[int(data_tp[i][4])])
            plt.vlines(left_label / fs, data_tp[i][2], data_tp[i][3], color_list[int(data_tp[i][4])])
            plt.vlines(right_label / fs, data_tp[i][2], data_tp[i][3], color_list[int(data_tp[i][4])])
            # 寻找有没有其他label
            

            plt.savefig("train_verify/" + recording_id + "_" + str(i) + ".png", bbox_inches="tight", pad_inches=0.0)
            plt.axis('off')
            plt.savefig("train_tran/" + recording_id + "_" + str(i) + ".png", bbox_inches="tight", pad_inches=0.0)
            # 图像大小 387 pixels x 154 pixels
            # Mel刻度: 0-512-1024-2048-4096-8192-16384
            # Log2(Mel刻度): 0-9-10-11-12-13-14
            # 纵轴为频率轴，遵循Mel刻度轴，0-9与9-14分开计算
            # 横轴为时间轴，对应坐标 (t / 10) / 387
            t_min = (left_label / fs) * 38.7
            t_max = (right_label / fs) * 38.7
            log_ymin = math.log2(data_tp[i][2])
            if log_ymin < 9:
                f_min = log_ymin / 9 * 154 / 6
            else:
                f_min = 154 / 6 + (log_ymin - 9) * 154 / 6
            log_ymax = math.log2(data_tp[i][3])
            if log_ymax < 9:
                f_max = log_ymax / 9 * 154 / 6
            else:
                f_max = 154 / 6 + (log_ymax - 9) * 154 / 6
            # label格式: png location \t t_min \t t_max \t f_min \t f_max \t species_id \t songtype_id
            file.write("./train_tran/" + recording_id + "_" + str(i) + ".png" + "\t" + str(t_min) +
                       "\t" + str(t_max) + "\t" + str(f_min) + "\t" + str(f_max)
                       + "\t" + str(data_tp[i][4]) + "\t" + str(data_tp[i][5]) + "\n")


if __name__ == '__main__':
    main()
