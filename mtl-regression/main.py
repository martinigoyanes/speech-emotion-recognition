import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from torch import nn
from utils import *
from SpeechDataset import *
from Network import *
from train import *
from experiment import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
dataset = "iemocap"
path = f"/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/code/data/{dataset}"

results = {
    "train": {
        "v_ccc": [],
        "a_ccc": [],
        "d_ccc": [],
        "ccc": [],
        "loss": [],
        "mse": [],
        "r2": [],
        "mae": [],
    },
    "val": {
        "v_ccc": [],
        "a_ccc": [],
        "d_ccc": [],
        "ccc": [],
        "loss": [],
        "mse": [],
        "r2": [],
        "mae": [],
    },
}
data = {
    "vocab_size": 2913,  # msp: 26590  iemocap: 2913 or 3438 
    "audio_feat": "paa+compare.npy",
    "text_feat": "text_seq_73_lemmas.npy", 
    "labels": "dimension.npy",
    "embeddings": f"{path}/embeddings_lemmas.npy", 
    "ratio": {
        "train": 0.65,
        "val": 0.15, 
    },
}
params = {
    "name": "mtl-regression",
    "audio_net": {
        "timesteps": 1,
        "feature_size": 198,
        "hidden_size": (256, 256, 256),
        "dropout": 0.3,
    },
    "text_net": {
        "embed_dim": 300,
        "timesteps": 73,
        "hidden_size": (256, 256, 256),
        "output_size": 64,
        "dropout": 0.3,
    },
    "net": {"hidden_size": (64, 32), "dropout": 0.4},
}

learning = {
    "test_bsz": 2048,  
    "train_bsz": 64,  
    "lr": 0.001,
    "scheduler": {"step": 7, "gamma": 0.1},
    "loss": {"alpha": 0.7, "beta": 0.2, "gamma": 0.1},
    "epochs": 100,  
    "patience": 10,  
    "delta": 0.0001,
    "seed": None, 
}
num_runs = 1
criterion = CCCLoss(
    learning["loss"]["alpha"],
    learning["loss"]["beta"],
    learning["loss"]["gamma"],
).to(device)

experiment(learning, params, data, device, path, criterion, num_runs=num_runs)
