import csv
import numpy as np

with open("metadata.csv") as metadata_f:
    csv_reader = csv.reader(metadata_f, delimiter=",")
    line_count = 0
    rows = []
    dimensions = np.empty((0, 3), float)
    for row in csv_reader:
        if line_count > 0:
            curr = np.array([[row[4], row[5], row[6]]], float)
            dimensions = np.append(dimensions, curr, axis=0)
        line_count += 1
np.save("dimension.npy", dimensions)
