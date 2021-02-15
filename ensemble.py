import pandas as pd


def ensemble(DataFrame):
    for i in range(24):
        name = "s" + str(i)
        for j in range(len(DataFrame) - 1):
            DataFrame[0][name] += DataFrame[j + 1][name]
        DataFrame[0][name] = 1 / len(DataFrame) * DataFrame[0][name]
    return DataFrame[0]


workdir = "D://RainYQ/rfcx-species-audio-detection/result/"
DataFrame = []
for i in range(9):
    if i == 0 or i == 1:
        pass
    else:
        DataFrame.append(pd.read_csv(workdir + "submission_" + str(i) + ".csv"))
ensemble(DataFrame).to_csv(workdir + "submission_ensemble.csv", index=False)
