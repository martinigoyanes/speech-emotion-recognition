import csv
import numpy as np

iemocap = {
    "ang": 0,
    "hap": 1,
    "sad": 2,
    "neu": 3,
    "fru": 4,
    "exc": 5,
    "fea": 6,
    "sur": 7,
    "dis": 8,  # there is 2 in the whole dataset
    "oth": 9,  # there is 2 in the whole dataset
    "xxx": np.nan,
}
msp = {
    "A": 0,
    "H": 1,
    "S": 2,
    "N": 3,
    "C": 4,
    "F": 5,
    "U": 6,
    "D": 7,  
    "O": 8,  
    "X": np.nan,
}
dataset = 'msp'
root_path = f"/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/code/data/{dataset}"

if dataset == 'iemocap':
    labels = iemocap
    emo_pos = 3
else:
    labels = msp
    emo_pos = 1

with open(f"{root_path}/metadata.csv") as metadata_f:
    csv_reader = csv.reader(metadata_f, delimiter=",")
    line_count = 0
    rows = []
    emotions = np.array([], int)
    for row in csv_reader:
        if line_count > 0:
            emo = [labels[row[emo_pos]]]
            emotions = np.append(emotions, emo, axis=0)
        line_count += 1
np.save(f"emotions.npy", emotions)
