import pandas as pd
import numpy as np

train_audio_path = "./train"
test_audio_path = "./test"
train_tp = "./train_tp.csv"
train_fp = "./train_fp.csv"
sample_submission = "./sample_submission.csv"

train_data_tp = pd.read_csv(train_tp)
train_data_fp = pd.read_csv(train_fp)

recording_ids = list(set(train_data_tp["recording_id"]))
print("train_tp含有的标记数目: ", len(train_data_tp))
print("train_tp含有的recording_id数目: ", len(recording_ids))
print("重复标记recording_id及数目: ")
q = 0
for recording_id in recording_ids:
    data = train_data_tp.loc[train_data_tp["recording_id"] == recording_id]
    if len(data) > 1:
        print(recording_id, "\t", len(data))
        q += 1
        t = list(np.array(data[["t_min","t_max","species_id"]]))
        print(t)
print("重复标记的recording_id数目: ", q)
# 不同标记具有重叠的时间段, 需要处理

