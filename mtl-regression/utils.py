from torch import nn
import numpy as np
import torch


class CCCLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(CCCLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs, labels):
        v_ccc, a_ccc, d_ccc = (
            self.ccc(outputs[:, 0], labels[:, 0]),
            self.ccc(outputs[:, 1], labels[:, 1]),
            self.ccc(outputs[:, 2], labels[:, 2]),
        )
        v_loss, a_loss, d_loss = 1.0 - v_ccc, 1.0 - a_ccc, 1.0 - d_ccc

        ccc = v_ccc + a_ccc + d_ccc
        loss = self.alpha * v_loss + self.beta * a_loss + self.gamma * d_loss
        # loss = (v_loss + a_loss + d_loss) / 3

        return loss, ccc, v_ccc, a_ccc, d_ccc

    def ccc(self, outputs, labels):
        labels_mean = torch.mean(labels)
        outputs_mean = torch.mean(outputs)
        covariance = (labels - labels_mean) * (outputs - outputs_mean)

        label_var = torch.mean(torch.square(labels - labels_mean))
        outputs_var = torch.mean(torch.square(outputs - outputs_mean))

        ccc = (2.0 * covariance) / (
            label_var + outputs_var + torch.square(labels_mean - outputs_mean)
        )
        return torch.mean(ccc)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, name="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            name (str): name for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mean_ccc_min = np.Inf
        self.delta = delta
        self.name = name
        self.trace_func = trace_func

    def __call__(self, mean_ccc, model):

        score = mean_ccc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(mean_ccc, model)
        elif np.abs(score) - np.abs(self.best_score) > self.delta:
            self.best_score = score
            self.save_checkpoint(mean_ccc, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, mean_ccc, model):
        """Saves model when validation loss decrease."""
        import copy

        if self.verbose:
            self.trace_func(
                f"Validation CCC increased ({self.mean_ccc_min:.6f} --> {mean_ccc:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.name)
        self.mean_ccc_min = mean_ccc


def plot_training(results, name):
    import matplotlib.pyplot as plt

    for phase in ["train", "val"]:
        v_loss = [1 - r for r in results[phase]["v_ccc"]]
        a_loss = [1 - r for r in results[phase]["a_ccc"]]
        d_loss = [1 - r for r in results[phase]["d_ccc"]]
        # summarize history for ccc
        plt.figure(figsize=(10, 12))
        ax0 = plt.subplot(2, 1, 1)
        plt.plot(
            results[phase]["v_ccc"], color="green", linestyle="dotted", label="v_ccc"
        )
        plt.plot(
            results[phase]["a_ccc"], color="blue", linestyle="dotted", label="a_ccc"
        )
        plt.plot(
            results[phase]["d_ccc"], color="red", linestyle="dotted", label="d_ccc"
        )
        plt.plot(results[phase]["ccc"], color="black", label="ccc")
        # plt.title(phase)
        min_val = np.min([results[phase]["ccc"], results[phase]["v_ccc"], results[phase]["a_ccc"],results[phase]["d_ccc"],])
        max_val = np.max([results[phase]["ccc"], results[phase]["v_ccc"], results[phase]["a_ccc"],results[phase]["d_ccc"],])
        ax0.yaxis.set_ticks(np.arange(min_val, max_val, 0.05))
        plt.xlabel("Epoch")
        plt.ylabel("CCC")
        plt.legend()

        ax1 = plt.subplot(2, 1, 2)
        plt.plot(v_loss, color="green", linestyle="dotted", label="v_loss")
        plt.plot(a_loss, color="blue", linestyle="dotted", label="a_loss")
        plt.plot(d_loss, color="red", linestyle="dotted", label="d_loss")
        plt.plot(results[phase]["loss"], color="black", label="loss")
        # plt.title(phase)
        min_val = np.min([results[phase]["loss"], v_loss, a_loss, d_loss])
        max_val = np.max([results[phase]["loss"], v_loss, a_loss, d_loss])
        ax1.yaxis.set_ticks(np.arange(min_val, max_val, 0.05))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig(f"{name}-{phase}.png")

def pretty_print(all, train, val, test):
    print("-" * 50)
    print("\t\t\tDATA")
    print("-" * 50)
    print("All data:")
    all.pretty_shape()
    print("-" * 10)
    print("Train data:")
    train.pretty_shape()
    print("-" * 10)
    print("Validation data:")
    val.pretty_shape()
    print("-" * 10)
    print("Test data:")
    test.pretty_shape()


def test_model(model, criterion, test_dataloader, data, device):
    import time

    since = time.time()
    model.eval()  # Ponemos el modelo en modo evaluación

    # Creamos las variables que almacenarán las salidas y las etiquetas
    running_mae = (
        running_r2
        ) = (    
        running_mse 
        ) = (    
        running_loss
        ) = running_ccc = running_v_ccc = running_a_ccc = running_d_ccc = 0.0

    # Iteramos sobre los datos.
    for sample in test_dataloader:
        audio_input = sample["audio"].to(device)
        text_input = sample["text"].to(device)
        labels = sample["labels"].to(device)

        # Tamaño del batch
        batchSize = labels.size(0)
        hidden = model.init_hidden()

        # Paso forward
        with torch.torch.no_grad():
            outputs = model(audio_input, text_input)
            model.repackage_hidden()
            loss, ccc, v_ccc, a_ccc, d_ccc = criterion(
                outputs, labels
            )
            mse = nn.MSELoss()(outputs, labels)
            mae = nn.L1Loss()(outputs, labels)
            r2 = r2_loss(outputs, labels)

        # Sacamos estadísticas
        running_loss += loss.item() * batchSize
        running_mse += mse.item() * batchSize
        running_mae += mae.item() * batchSize
        running_r2 += r2.item() * batchSize
        running_ccc += ccc.item() * batchSize
        running_v_ccc += v_ccc.item() * batchSize
        running_a_ccc += a_ccc.item() * batchSize
        running_d_ccc += d_ccc.item() * batchSize

    # Loss acumulada en la época
    epoch_loss = running_loss / len(data)
    epoch_mse = running_mse / len(data)
    epoch_mae = running_mae / len(data)
    epoch_r2 = running_r2 / len(data)
    epoch_ccc = running_ccc / len(data) / 3
    epoch_v_ccc = running_v_ccc / len(data)
    epoch_a_ccc = running_a_ccc / len(data)
    epoch_d_ccc = running_d_ccc / len(data)


    time_elapsed = time.time() - since
    print(
        "Test R2: {:.4f} MAE: {:.4f} MSE: {:.4f} loss: {:.4f}\tCCC: {:.4f}  v_ccc: {:.4f}  a_ccc: {:.4f}  d_ccc: {:.4f} ".format(
            epoch_r2,
            epoch_mae,
            epoch_mse,
            epoch_loss,
            epoch_ccc,
            epoch_v_ccc,
            epoch_a_ccc,
            epoch_d_ccc,
        )
    )
    print(
        "Testing complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    return (
        epoch_mse,
        epoch_mae,
        epoch_r2,
        epoch_ccc,
        epoch_v_ccc,
        epoch_a_ccc,
        epoch_d_ccc
    )

def dry_run(model, criterion, test_dataloader, data, device):
    import time

    since = time.time()
    model.eval()  # Ponemos el modelo en modo evaluación

    # Creamos las variables que almacenarán las salidas y las etiquetas
    running_mae = (
        running_r2
        ) = (    
        running_mse 
        ) = (    
        running_loss
        ) = running_ccc = running_v_ccc = running_a_ccc = running_d_ccc = 0.0

    outputs_ = []
    labels_ = []
    # Iteramos sobre los datos.
    for sample in test_dataloader:
        audio_input = sample["audio"].to(device)
        text_input = sample["text"].to(device)
        labels = sample["labels"].to(device)

        # Tamaño del batch
        batchSize = labels.size(0)
        hidden = model.init_hidden()

        # Paso forward
        with torch.torch.no_grad():
            outputs = model(audio_input, text_input)
            model.repackage_hidden()
            loss, ccc, v_ccc, a_ccc, d_ccc = criterion(
                outputs, labels
            )
            mse = nn.MSELoss()(outputs, labels)
            mae = nn.L1Loss()(outputs, labels)
            r2 = r2_loss(outputs, labels)

        for i in range(len(outputs)):
            outputs_.append([ o.item() for o in outputs[i]])
            labels_.append([ l.item() for l in labels[i]])

        # Sacamos estadísticas
        running_loss += loss.item() * batchSize
        running_mse += mse.item() * batchSize
        running_mae += mae.item() * batchSize
        running_r2 += r2.item() * batchSize
        running_ccc += ccc.item() * batchSize
        running_v_ccc += v_ccc.item() * batchSize
        running_a_ccc += a_ccc.item() * batchSize
        running_d_ccc += d_ccc.item() * batchSize

    # Loss acumulada en la época
    epoch_loss = running_loss / len(data)
    epoch_mse = running_mse / len(data)
    epoch_mae = running_mae / len(data)
    epoch_r2 = running_r2 / len(data)
    epoch_ccc = running_ccc / len(data) / 3
    epoch_v_ccc = running_v_ccc / len(data)
    epoch_a_ccc = running_a_ccc / len(data)
    epoch_d_ccc = running_d_ccc / len(data)

    time_elapsed = time.time() - since
    print(
        "Test R2: {:.4f} MAE: {:.4f} MSE: {:.4f} loss: {:.4f}\tCCC: {:.4f}  v_ccc: {:.4f}  a_ccc: {:.4f}  d_ccc: {:.4f} ".format(
            epoch_r2,
            epoch_mae,
            epoch_mse,
            epoch_loss,
            epoch_ccc,
            epoch_v_ccc,
            epoch_a_ccc,
            epoch_d_ccc,
        )
    )
    print('Prediction\t|\tGround Truth')
    for i in range(len(outputs_[:30])):
        out = [f'{o:.2f}' for o in outputs_[i]]
        print(f'{out}\t|\t{labels_[i]}')
    print(
        "Testing complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    # r2 = 1 - ss_res / ss_tot
    r2 = ss_res / ss_tot
    return r2