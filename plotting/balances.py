import matplotlib.pyplot as plt
import numpy as np
import matplotlib


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
   0: ("Angry", 'r'),
   1: ("Happy", 'g'),
   2: ("Sad", 'b'),
   3: ("Neutral", 'pink'),
   4: ("Comtempt", 'm'),
   5: ("Fear", 'y'),
   6: ("Surprise", 'k'),
   7: ("Disgust", 'c'),
   8: ("Other", 'lime'),
}

def clean_data(X, y, classes):
    # Cleaning the data
    mapping = {}
    ids = {}
    classes = classes.copy()
    for k, v in classes.items():
        emo_name = v[0]
        emo_num = k
        mapping[emo_name] = emo_num

    for emo in ['Other', 'Frustration', 'Disgust']:
        ids[emo] = np.argwhere(y == mapping[emo])
        y = np.delete(y, ids[emo], 0)
        X = np.delete(X, ids[emo], 0)
        del classes[mapping[emo]]

    # merge excitement with happiness
    ids["Excited"] = np.argwhere(y == mapping["Excited"])
    y[ids["Excited"]] = mapping["Happy"]
    X[ids["Excited"]] = mapping["Happy"]
    del classes[mapping["Excited"]]


    # reorganize dict so classes are in order
    i = 0
    new_classes = {}
    for k,v in classes.items():
        new_classes[i] = v 
        if v[0] == 'Fear' or v[0] == 'Surprise':
            ids[v[0]] = np.argwhere(y == mapping[v[0]])
            y[ids[v[0]]] = i
        i += 1
    

    return X, y, new_classes


def load_data(root_path, tipo='classes'):
    input = np.load(f"{root_path}/dimension.npy")
    labels = np.load(f"{root_path}/emotions.npy")

    if tipo == 'classes':
        nan_idx = np.argwhere(np.isnan(labels))
        labels = np.delete(labels, nan_idx, 0)
        labels = labels.astype(int)
        input = np.delete(input, nan_idx, 0)

    return input, labels


def get_counts(samples, classes):
    # counts = { k: len(np.argwhere(samples==k)) for k,v in dic.items()} Not very readable
    counts = {k: 0 for k, v in classes.items()}
    total_count = 0
    for x in samples:
        if x in counts.keys():
            counts[x] = counts[x] + 1
            total_count = total_count + 1

    return counts, total_count


def plot_classes(emo, classes, tipo='default'):
    ####### DISCRETE CLASSES ############

    counts, tot_count = get_counts(emo, classes)

    class_names = [v[0] for k, v in classes.items()]
    colors = [v[1] for k,v in classes.items()]

    class_index = [k for k in counts.keys()]
    counts = [v for v in counts.values()]

    plt.figure(figsize=(15, 8))
    graph = plt.bar(class_index, counts, align="center", color=colors)
    plt.xticks(class_index, class_names)
    plt.ylabel("N Samples")

    i = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(
            x + width / 2,
            y + height * 1.01,
            f"{(counts[i] / tot_count) * 100:.2f}%",
            ha="center",
            weight="bold",
        )
        i += 1

    plt.savefig(f"/export/usuarios01/miglesias/tfg/scripts/plotting/plots/balance-emos-{tipo}-{dataset}")

def plot_vad(vad, tipo='default'):
    colors = {
        'v': 'red',
        'a': 'blue',
        'd': 'green'
    }
    vad = {'Valence': (vad[:, 0], 'red'),
           'Arousal': (vad[:, 1], 'blue'),
           'Dominance': (vad[:, 2], 'green')
           }

    for name, v in vad.items():
        plt.figure(figsize=(8, 6))
        x = v[0]
        color = v[1]
        n, bins, pathces = plt.hist(x, bins=5, density=False, color=color)

        plt.grid(True)
        plt.xlabel(name)
        plt.ylabel('N Samples')
        plt.grid(True)

        tot_count = np.sum(n)
        for i in range(len(n)):
            x = (bins[i] + bins[i+1])/2
            y = n[i]
            plt.text(
                x,
                y * 1.01,
                f"{(y / tot_count) * 100:.2f}%",
                ha="center",
                weight="bold",
            )

        plt.savefig(f"/export/usuarios01/miglesias/tfg/scripts/plotting/plots/balance-{name}-{tipo}-{dataset}")

if __name__ == "__main__":
    dataset = "msp"
    root_path = f"/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/code/data/{dataset}"

    if dataset == "iemocap":
        globclasses = iemocap
    else:
        globclasses = msp

    ## Plot default classes
    vad, emo = load_data(root_path, tipo='classes')
    plot_classes(emo, globclasses, tipo='default')

    ## Plot clean classes
    if dataset == 'iemocap':
        vad, emo = load_data(root_path, tipo='classes')
        vad, emo, new_classes = clean_data(vad, emo, globclasses)
        plot_classes(emo, new_classes, tipo='clean')
    
    ## Plot default vad
    vad, emo = load_data(root_path, tipo='vad')
    plot_vad(vad, tipo='default')


    ## Plot clean vad
    if dataset == 'iemocap':
        vad, emo = load_data(root_path, tipo='classes') # tipo='classes' since we want to clean out the nans
        vad, emo, new_classes = clean_data(vad, emo, globclasses)
        plot_vad(vad, tipo='clean')
