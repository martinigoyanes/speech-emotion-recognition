import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader


class SpeechDataset(Dataset):
    """Dermoscopy dataset."""

    def __init__(
        self, audio_feat, text_feat, labels, text_steps, audio_steps, root_path=None
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
        self.labels = (
            np.load(f"{root_path}/{labels}") if isinstance(labels, str) else labels
        )
        self.classes = ["v", "a", "d"]

        self.text_steps = text_steps
        self.audio_steps = audio_steps

    def __len__(self):
        return len(self.audio_feat)

    def __getitem__(self, idx):
        audio_sample = self.audio_feat[idx]
        text_sample = self.text_feat[idx]
        labels_sample = self.labels[idx]

        sample = {"audio": audio_sample, "text": text_sample, "labels": labels_sample}

        return sample

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
        if np.max(self.labels) == 5.5:  # case of iemocap
            self.labels = np.where(self.labels > 5.0, 5.0, self.labels)
            self.labels = np.where(self.labels < 1.0, 1.0, self.labels)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(self.labels)
        self.labels = scaler.transform(self.labels)

        # Text features
        self.text_feat = self.text_feat.reshape(
            (self.text_feat.shape[0], self.text_steps)
        )

        self.to_tensor()

    def to_tensor(self):
        self.audio_feat = torch.from_numpy(self.audio_feat).type(torch.FloatTensor)
        self.text_feat = torch.from_numpy(self.text_feat).type(torch.IntTensor)
        self.labels = torch.from_numpy(self.labels).type(torch.FloatTensor)

    def to_numpy(self):
        self.audio_feat = self.audio_feat.numpy()
        self.text_feat = self.text_feat.numpy()
        self.labels = self.labels.numpy()

    def pretty_shape(self):
        print(
            f"Audio features: {self.audio_feat.size()}\nText features: {self.text_feat.size()}\nLabels: {[self.labels.size()]}"
        )


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
    # train_data = data[:train_len]
    # # val_data = data[train_len : train_len + val_len]
    # # test_data = data[train_len + val_len :]
    # test_data = data[train_len : train_len + val_len]
    # val_data = data[train_len + val_len :]

    train_data = SpeechDataset(
        train_data["audio"],
        train_data["text"],
        train_data["labels"],
        text_steps,
        audio_steps,
    )
    val_data = SpeechDataset(
        val_data["audio"], val_data["text"], val_data["labels"], text_steps, audio_steps
    )
    test_data = SpeechDataset(
        test_data["audio"],
        test_data["text"],
        test_data["labels"],
        text_steps,
        audio_steps,
    )

    return train_data, val_data, test_data


def load_data(
    audio_feat,
    text_feat,
    labels,
    root_path,
    text_steps,
    audio_steps,
    train_ratio,
    val_ratioi,
    train_bsz,
    test_bsz,
):
    data = SpeechDataset(
        audio_feat,
        text_feat,
        labels,
        text_steps=text_steps,
        audio_steps=audio_steps,
        root_path=root_path,
    )
    data.scale()

    train_data, val_data, test_data = split_data(
        data=data,
        train_ratio=0.65,
        val_ratio=0.15,
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
