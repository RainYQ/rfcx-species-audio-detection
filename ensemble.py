import pandas as pd


def ensemble(DataFrame1, DataFrame2, DataFrame3):
    for i in range(24):
        name = "s" + str(i)
        DataFrame1[name] = 1 / 3 * (DataFrame1[name] + 1 / 3 * DataFrame2[name] + 1 / 3 * DataFrame3[name])
    return DataFrame1


workdir = "D://RainYQ/rfcx-species-audio-detection/result/"
DataFrame1 = pd.read_csv(workdir + "submission_0.csv")
DataFrame2 = pd.read_csv(workdir + "submission_1.csv")
DataFrame3 = pd.read_csv(workdir + "submission_2.csv")
ensemble(DataFrame1, DataFrame2, DataFrame3).to_csv(workdir + "submission_ensemble.csv", index=False)
