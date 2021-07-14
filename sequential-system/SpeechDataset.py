import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader


class SpeechDataset(Dataset):
    """Dermoscopy dataset."""
    
    def __init__(
        self, classes, audio_feat, text_feat, labels_cat, labels_dim, text_steps, audio_steps, root_path=None
    ):
        """
        Args:
            audio_feat (string): Nombre fichero con las audio features.
            text_feat (string): Nombre fichero con las text features.
            labels (string): Nombre fichero con las labels (valence, arousal, dominance).
            root_path (string): Path del directorio de trabajo
        """
        self.audio_feat = (
            np.load(f"{root_path}/{audio_feat}")
            if isinstance(audio_feat, str)
            else audio_feat
        )
        self.text_feat = (
            np.load(f"{root_path}/{text_feat}")
            if isinstance(text_feat, str)
            else text_feat
        )
        self.labels_cat = (
            np.load(f"{root_path}/{labels_cat}") if isinstance(labels_cat, str) else labels_cat
        )
        self.labels_dim = (
            np.load(f"{root_path}/{labels_dim}") if isinstance(labels_dim, str) else labels_dim
        )
        self.classes = classes.copy()

        self.text_steps = text_steps
        self.audio_steps = audio_steps
        self.imbalances = {}
        for k,v in self.classes.items():
            self.imbalances[v] = 0

    def __len__(self):
        return len(self.audio_feat)

    def __getitem__(self, idx):
        audio_sample = self.audio_feat[idx]
        text_sample = self.text_feat[idx]
        labels_cat_sample = self.labels_cat[idx]
        labels_dim_sample = self.labels_dim[idx]

        sample = {"audio": audio_sample, "text": text_sample, "labels_cat": labels_cat_sample, "labels_dim": labels_dim_sample}

        return sample
    def drop_nan(self):
        if torch.is_tensor(self.labels_cat):
            self.to_numpy()

        nan_idx = np.argwhere(np.isnan(self.labels_cat))
        self.audio_feat = np.delete(self.audio_feat, nan_idx, 0)
        self.text_feat = np.delete(self.text_feat, nan_idx, 0)
        self.labels_cat = np.delete(self.labels_cat, nan_idx, 0)
        self.labels_dim = np.delete(self.labels_dim, nan_idx, 0)

        self.to_tensor()

    def oversample(self):
        #! ISSUE: Can not do resample of continuous labels
        from imblearn.over_sampling import SVMSMOTE

        if torch.is_tensor(self.labels_cat):
            self.to_numpy()
        X = np.concatenate((self.audio_feat, self.text_feat), axis=1)
        y = np.concatenate((self.labels_cat.reshape(self.labels_cat.shape[0], 1), self.labels_dim), axis=1)
        oversample = SVMSMOTE(sampling_strategy={6: 1000, 7:1000})
        X, y = oversample.fit_resample(X, y)

        self.audio_feat = X[:, :198]
        self.text_feat = X[:, 198:]
        self.labels_cat = y[:, :1]
        self.labels_dim = y[:, 1:]
        self.to_tensor()


    def clean_emos(self, emotions):
        tmpclasses = {v:k for k,v in self.classes.items()}
        self.to_numpy()

        for emo in emotions:
            ids = np.argwhere( self.labels_cat == tmpclasses[emo])
            if emo == 'exc':
                # merge excitement with happiness
                self.audio_feat[ids] =  tmpclasses['hap']
                self.text_feat[ids] = tmpclasses['hap']
                self.labels_cat[ids] =  tmpclasses['hap']
                self.labels_dim[ids] =  tmpclasses['hap']
            else:
                self.audio_feat = np.delete( self.audio_feat, ids, 0)
                self.text_feat = np.delete( self.text_feat, ids, 0)
                self.labels_cat = np.delete( self.labels_cat, ids, 0)
                self.labels_dim = np.delete( self.labels_dim, ids, 0)

            del self.classes[tmpclasses[emo]]

        i = 0
        new_classes = {}
        for key in self.classes.keys():
            new_classes[key] = i
            i += 1 

        for idx in range(len(self.labels_cat)):
            self.labels_cat[idx] = new_classes[self.labels_cat[idx]]

        i = 0
        new_classes = {}
        for v in self.classes.values():
            new_classes[i] = v
            i += 1 
        self.classes = new_classes
        self.to_tensor()

    def check_imbalances(self):
        if torch.is_tensor(self.audio_feat):
            self.to_numpy()       

        for label in self.labels_cat:
            name = self.classes[label]
            self.imbalances[name] += 1

        for k,v in self.imbalances.items():
            prop = self.imbalances[k]/self.__len__()
            self.imbalances[k] = (prop, self.imbalances[k])

        self.to_tensor()

    def scale(self):
        if torch.is_tensor(self.audio_feat):
            self.to_numpy()

        # Audio features
        scaler = StandardScaler()
        if len(self.audio_feat.shape) == 2:
            self.audio_feat = np.reshape(
                self.audio_feat,
                (self.audio_feat.shape[0], self.audio_steps, self.audio_feat.shape[1]),
            )

        scaler = scaler.fit(
            self.audio_feat.reshape(
                self.audio_feat.shape[0],
                self.audio_feat.shape[1] * self.audio_feat.shape[2],
            )
        )
        scaled_feat = scaler.transform(
            self.audio_feat.reshape(
                (
                    self.audio_feat.shape[0],
                    self.audio_feat.shape[1] * self.audio_feat.shape[2],
                )
            )
        )
        self.audio_feat = scaled_feat.reshape(
            (
                self.audio_feat.shape[0],
                self.audio_feat.shape[1],
                self.audio_feat.shape[2],
            )
        )

        # Labels
        # remove outlier, < 1, > 5
        if np.max(self.labels_dim) == 5.5:
            self.labels_dim = np.where(self.labels_dim == 5.5, 5.0, self.labels_dim)
            self.labels_dim = np.where(self.labels_dim == 0.5, 1.0, self.labels_dim)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(self.labels_dim)
        self.labels_dim = scaler.transform(self.labels_dim)


        # Text features
        self.text_feat = self.text_feat.reshape(
            (self.text_feat.shape[0], self.text_steps)
        )

        self.to_tensor()

    def to_tensor(self):
        self.audio_feat = torch.from_numpy(self.audio_feat).type(torch.FloatTensor)
        self.text_feat = torch.from_numpy(self.text_feat).type(torch.IntTensor)
        self.labels_cat = torch.from_numpy(self.labels_cat).type(torch.LongTensor)
        self.labels_dim = torch.from_numpy(self.labels_dim).type(torch.FloatTensor)

    def to_numpy(self):
        self.audio_feat = self.audio_feat.numpy()
        self.text_feat = self.text_feat.numpy()
        self.labels_cat = self.labels_cat.numpy()
        self.labels_dim = self.labels_dim.numpy()

    def pretty_shape(self):
        print(
            f"Audio features: {self.audio_feat.size()}\nText features: {self.text_feat.size()}\nLabels Cat: {[self.labels_cat.size()]}\nLabels Dim: {[self.labels_dim.size()]}"
        )
        imbalances_txt = ''
        for k,v in self.imbalances.items(): imbalances_txt += f'{k}: ({v[0]*100:.1f}%, {v[1]}), '
        print(imbalances_txt) 


def split_data(data, train_ratio, val_ratio, text_steps, audio_steps):
    train_len = int(train_ratio * len(data))
    val_len = int(val_ratio * len(data))
    test_len = int(len(data) - train_len - val_len)

    train_data, val_data, test_data = torch.utils.data.random_split(
        data, [train_len, val_len, len(data) - train_len - val_len]
    )
    train_data, val_data, test_data = (
        data[train_data.indices],
        data[val_data.indices],
        data[test_data.indices],
    )

    train_data = SpeechDataset(
        data.classes,
        train_data["audio"],
        train_data["text"],
        train_data["labels_cat"],
        train_data["labels_dim"],
        text_steps,
        audio_steps,
    )
    train_data.check_imbalances() 
    val_data = SpeechDataset(
        data.classes, val_data["audio"], val_data["text"], val_data["labels_cat"], val_data["labels_dim"],text_steps, audio_steps
    )
    val_data.check_imbalances() 
    test_data = SpeechDataset(
        data.classes,
        test_data["audio"],
        test_data["text"],
        test_data["labels_cat"],
        test_data["labels_dim"],
        text_steps,
        audio_steps,
    )
    test_data.check_imbalances()

    return train_data, val_data, test_data


def load_data(
    classes,
    audio_feat,
    text_feat,
    labels_cat,
    labels_dim,
    root_path,
    text_steps,
    audio_steps,
    train_ratio,
    val_ratio,
    train_bsz,
    test_bsz,
):
    data = SpeechDataset(
        classes,
        audio_feat,
        text_feat,
        labels_cat,
        labels_dim,
        text_steps=text_steps,
        audio_steps=audio_steps,
        root_path=root_path,
    )
    data.drop_nan()
    # data.oversample()
    # data.clean_emos(['fru', 'fea', 'sur', 'oth', 'dis', 'exc'])
    data.clean_emos(['fru', 'oth', 'dis', 'exc'])
    data.check_imbalances()
    data.scale()
    

    train_data, val_data, test_data = split_data(
        data=data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        text_steps=text_steps,
        audio_steps=audio_steps,
    )

    train_dataloader = DataLoader(
        train_data, batch_size=train_bsz, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_data, batch_size=test_bsz, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_data, batch_size=test_bsz, shuffle=False, num_workers=4
    )

    return {"all": data, "train": train_data, "val": val_data, "test": test_data}, {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }
