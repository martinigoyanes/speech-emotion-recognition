import matplotlib.pyplot as plt
import numpy as np
import matplotlib

dataset = 'msp'
root_path = f"/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/code/data/{dataset}"
mode = '3d'
num_points = 2000

iemocap = {
    0:'ang',
    1:'hap', 
    2:'sad', 
    3:'neu', 
    4:'fru', 
    5:'exc', 
    6:'fea', 
    7:'sur', 
}
msp = {
   0: ("Angry", 'r'),
   1: ("Happy", 'g'),
   2: ("Sad", 'b'),
   3: ("Neutral", 'pink'),
   4: ("Comtempt", 'm'),
   5: ("Fear", 'y'),
   6: ("Surprise", 'b'),
   7: ("Disgust", 'c'),
}
if dataset == 'iemocap':
    colors = iemocap
else:
    colors = msp

def load_data(root_path):
    input = np.load(f'{root_path}/dimension.npy')
    labels = np.load(f'{root_path}/emotions-drop.npy')
    
    nan_idx = np.argwhere(np.isnan(labels))
    labels = np.delete(labels, nan_idx, 0)
    labels = labels.astype(int)
    input = np.delete(input, nan_idx, 0)

    return input, labels

vad, emo = load_data(root_path)

if mode == '3d':
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    for class_num, class_name in colors.items():
        idx = np.where(emo==class_num)[0][:num_points]
        ax.scatter(vad[idx, 0], vad[idx, 2], vad[idx, 1], label=colors[class_num][0], c=colors[class_num][1])

    ax.set_xlabel('Valence')
    ax.set_ylabel('Dominance')
    ax.set_zlabel('Arousal')
    ax.legend()
    plt.savefig('emotions-3D.png')

if mode == '2d':
    fig = plt.figure(figsize=(40,10))

    # Valence vs Dominance
    ax0 = plt.subplot(1, 3, 1)
    for class_num, class_name in colors.items():
        idx = np.where(emo==class_num)[0][:num_points]
        ax0.scatter(vad[idx, 0], vad[idx, 2], label=colors[class_num][0], c=colors[class_num][1])
    ax0.set_xlabel('Valence')
    ax0.set_ylabel('Dominance')
    ax0.legend()
    plt.grid(True)

    # Valence vs Arousal
    ax1 = plt.subplot(1, 3, 2)
    for class_num, class_name in colors.items():
        idx = np.where(emo==class_num)[0][:num_points]
        ax1.scatter(vad[idx, 0], vad[idx, 1], label=colors[class_num][0], c=colors[class_num][1])
    ax1.set_xlabel('Valence')
    ax1.set_ylabel('Arousal')
    ax1.legend()
    plt.grid(True)


    # Arousal vs Dominance 
    ax2 = plt.subplot(1, 3, 3)
    for class_num, class_name in colors.items():
        idx = np.where(emo==class_num)[0][:num_points]
        ax2.scatter(vad[idx, 1], vad[idx, 2], label=colors[class_num][0], c=colors[class_num][1])
    ax2.set_xlabel('Arousal')
    ax2.set_ylabel('Dominance')
    ax2.legend()
    plt.grid(True)

    plt.savefig('emotions-2D.png')
