import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.cluster import KMeans

dataset = "iemocap"
root_path = f"/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/code/data/{dataset}"
num_points = 2000

iemocap = {
    0: ("Angry", "r"),
    1: ("Happy", "g"),
    2: ("Sad", "b"),
    3: ("Neutral", "pink"),
    4: ("Frustration", "m"),
    5: ("Excited", "y"),
    6: ("Fear", "k"),
    7: ("Surprise", "c"),
    8: ("Disgust", "lime"),
    9: ("Other", "tab:grey"),
}
msp = {
    0: ("Angry", "r"),
    1: ("Happy", "g"),
    2: ("Sad", "b"),
    3: ("Neutral", "pink"),
    4: ("Comtempt", "m"),
    5: ("Fear", "y"),
    6: ("Surprise", "k"),
    7: ("Disgust", "c"),
    8: ("Other", "lime"),
}
if dataset == "iemocap":
    colors = iemocap
else:
    colors = msp


def load_data(root_path):
    input = np.load(f"{root_path}/dimension.npy")
    labels = np.load(f"{root_path}/emotions.npy")

    nan_idx = np.argwhere(np.isnan(labels))
    labels = np.delete(labels, nan_idx, 0)
    labels = labels.astype(int)
    input = np.delete(input, nan_idx, 0)

    return input, labels


vad, emo = load_data(root_path)

X = vad
kmeans = KMeans(8, random_state=0)
labels = kmeans.fit(X).predict(X)
centers = kmeans.cluster_centers_

fig = plt.figure(figsize=(40, 15))

ax0 = plt.subplot(1, 3, 1)
for class_num, class_name in colors.items():
    idx = np.where(emo == class_num)[0][:100]
    ax0.scatter(
        X[idx, 0],
        X[idx, 2],
        alpha=0.1,
        label=colors[class_num][0],
        c=colors[class_num][1],
    )
    ax0.scatter(
        centers[class_num, 0],
        centers[class_num, 2],
        c=colors[class_num][1],
        marker="v",
        s=100,
    )
ax0.set_xlabel("valence")
ax0.set_ylabel("dominance")
ax0.grid(True)
ax0.legend()
for i, c in enumerate(centers):
    radio = np.linalg.norm(X[labels == i] - c, axis=1).max()
    ax0.add_patch(plt.Circle(c, radio, alpha=0.1, lw=3, color=colors[i][1]))

ax1 = plt.subplot(1, 3, 2)
for class_num, class_name in colors.items():
    idx = np.where(emo == class_num)[0][:100]
    ax1.scatter(
        X[idx, 0],
        X[idx, 1],
        alpha=0.1,
        label=colors[class_num][0],
        c=colors[class_num][1],
    )
    ax1.scatter(
        centers[class_num, 0],
        centers[class_num, 1],
        c=colors[class_num][1],
        marker="v",
        s=100,
    )
ax1.set_xlabel("valence")
ax1.set_ylabel("arousal")
ax1.grid(True)
ax1.legend()
for i, c in enumerate(centers):
    radio = np.linalg.norm(X[labels == i] - c, axis=1).max()
    ax1.add_patch(plt.Circle(c, radio, alpha=0.1, lw=3, color=colors[i][1]))

ax2 = plt.subplot(1, 3, 3)
for class_num, class_name in colors.items():
    idx = np.where(emo == class_num)[0][:100]
    ax2.scatter(
        X[idx, 1],
        X[idx, 2],
        alpha=0.1,
        label=colors[class_num][0],
        c=colors[class_num][1],
    )
    ax2.scatter(
        centers[class_num, 1],
        centers[class_num, 2],
        c=colors[class_num][1],
        marker="v",
        s=100,
    )
ax2.set_xlabel("arousal")
ax2.set_ylabel("dominance")
ax2.grid(True)
ax2.legend()
for i, c in enumerate(centers):
    radio = np.linalg.norm(X[labels == i] - c, axis=1).max()
    ax2.add_patch(plt.Circle(c, radio, alpha=0.1, lw=3, color=colors[i][1]))

plt.savefig("kmeans.png")
