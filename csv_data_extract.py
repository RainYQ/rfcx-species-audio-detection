import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

# 统计不同物种具有的标记数目
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
print(train_data_tp.groupby(['species_id']).size())
plt.hist(np.array(train_data_tp['species_id']), bins=24, alpha=0.7)
plt.xlabel("物种")
plt.ylabel("物种标记频数")
plt.title("物种标记频数分布直方图")
plt.show()
