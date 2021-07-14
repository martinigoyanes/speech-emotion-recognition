from torch import nn
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report

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
        self.mean_loss_max = - np.Inf # changed
        self.delta = delta
        self.name = name
        self.trace_func = trace_func

    def __call__(self, mean_loss, model):

        score = mean_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(mean_loss, model)
        # elif np.abs(self.best_score) - np.abs(score) > self.delta:
        elif np.abs(score) - np.abs(self.best_score)> self.delta:
            self.best_score = score
            self.save_checkpoint(mean_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, mean_loss, model):
        """Saves model when validation loss decrease."""
        import copy

        if self.verbose:
            self.trace_func(
                # f"Validation Total Loss decreased ({self.mean_loss_max:.6f} --> {mean_loss:.6f}).  Saving model ..."
                f"Validation Accuracy increased ({self.mean_loss_max:.6f} --> {mean_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.name)
        self.mean_loss_max = mean_loss


def plot_training(results, name):
    import matplotlib.pyplot as plt

    for phase in ["train", "val"]:
        # summarize history for ccc
        plt.figure(figsize=(15, 12))
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
        plt.plot(results[phase]["ccc"], color="yellow", linestyle="dotted", label="ccc")
        plt.plot(results[phase]["acc"], color="black", label="accuracy")
        # plt.title(phase)
        min_val = np.min([results[phase]["ccc"], results[phase]["v_ccc"], results[phase]["a_ccc"],results[phase]["d_ccc"],results[phase]["acc"]])
        max_val = np.max([results[phase]["ccc"], results[phase]["v_ccc"], results[phase]["a_ccc"],results[phase]["d_ccc"],results[phase]["acc"]])
        ax0.yaxis.set_ticks(np.arange(min_val, max_val, 0.05))
        plt.xlabel("Epoch")
        plt.ylabel("CCC")
        plt.legend()

        ax1 = plt.subplot(2, 1, 2)
        plt.plot(results[phase]["cat_loss"], color="green", linestyle="dotted", label="classification_loss")
        plt.plot(results[phase]["dim_loss"], color="blue", linestyle="dotted", label="regression_loss")
        plt.plot(results[phase]["tot_loss"], color="black", label="total_loss")
        # plt.title(phase)
        min_val = np.min([results[phase]["cat_loss"], results[phase]["dim_loss"], results[phase]["tot_loss"],results[phase]["d_ccc"],results[phase]["acc"]])
        max_val = np.max([results[phase]["cat_loss"], results[phase]["dim_loss"], results[phase]["tot_loss"],results[phase]["d_ccc"],results[phase]["acc"]])
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


def test_model(model, crit_cat, crit_dim, test_dataloader, data, device):
    import time

    since = time.time()
    model.eval()  # Ponemos el modelo en modo evaluación

    # Creamos las variables que almacenarán las salidas y las etiquetas
    running_tot_loss = running_cat_loss = running_dim_loss = \
     running_acc = running_ccc = running_v_ccc = running_a_ccc = running_d_ccc = 0.0
    outputs_ = []
    labels_ = []
    # Iteramos sobre los datos.
    for sample in test_dataloader:
        audio_input = sample["audio"].to(device)
        text_input = sample["text"].to(device)
        labels_cat = sample["labels_cat"].to(device)
        labels_dim = sample["labels_dim"].to(device)

        # Tamaño del batch
        batchSize = labels_cat.size(0)
        hidden = model.init_hidden()

        # Paso forward
        with torch.torch.no_grad():
            out_cat, out_dim = model(audio_input, text_input)
            model.repackage_hidden()
            dim_loss, ccc, v_ccc, a_ccc, d_ccc = crit_dim(
                out_dim, labels_dim
            )
            cat_loss = crit_cat(out_cat, labels_cat)

            tot_loss = 0.7*cat_loss + 0.3*dim_loss
            tot_loss = cat_loss + dim_loss

            out_cat=nn.functional.softmax(out_cat.data,dim=1)
            indices, preds = torch.max(out_cat, dim=1)
            # print('Predicted\t|\tGround Truth')
            # for i in range(len(preds[:100])):
            #     yhat = data.classes[preds[i].item()]
            #     ytrue = data.classes[labels_cat[i].item()]
            #     print(f'{yhat}\t|\t{ytrue}')

            for i in range(len(preds)):
                outputs_.append(preds[i].item())
                labels_.append(labels_cat[i].item())
            acc = accuracy_score(outputs_, labels_)
            # Sacamos estadísticas
            running_tot_loss += tot_loss.item() * batchSize
            running_cat_loss += cat_loss.item() * batchSize
            running_dim_loss += dim_loss.item() * batchSize
            running_ccc += ccc.item() * batchSize
            running_acc += acc.item() * batchSize
            running_v_ccc += v_ccc.item() * batchSize
            running_a_ccc += a_ccc.item() * batchSize
            running_d_ccc += d_ccc.item() * batchSize

    # Loss acumulada en la época
    epoch_tot_loss = running_tot_loss / len(data)
    epoch_cat_loss = running_cat_loss / len(data)
    epoch_dim_loss = running_dim_loss / len(data)
    epoch_ccc = running_ccc / len(data) / 3
    epoch_acc = running_acc / len(data)
    epoch_v_ccc = running_v_ccc / len(data)
    epoch_a_ccc = running_a_ccc / len(data)
    epoch_d_ccc = running_d_ccc / len(data)

    print(
        "Test Tot-loss: {:.4f} Cat-loss: {:.4f} Dim-loss: {:.4f} \tACC: {:.4f}  CCC: {:.4f}  v_ccc: {:.4f}  a_ccc: {:.4f}  d_ccc: {:.4f} ".format(
            epoch_tot_loss,
            epoch_cat_loss,
            epoch_dim_loss,
            epoch_acc,
            epoch_ccc,
            epoch_v_ccc,
            epoch_a_ccc,
            epoch_d_ccc,
        )
    )
    time_elapsed = time.time() - since
    label_names = [ v for k,v in data.classes.items()]
    print_confusion_matrix(outputs_, labels_, label_names)
    print(
        "Testing complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    return (
        epoch_tot_loss,
        epoch_cat_loss,
        epoch_dim_loss,
        epoch_acc,
        epoch_ccc,
        epoch_v_ccc,
        epoch_a_ccc,
        epoch_d_ccc,
    )

def print_confusion_matrix(outputs_, labels_, label_names):
    title = ''
    column = ''
    for name in label_names: 
        title += f'{name}\t'
        column += f'{name}'
    confusions = confusion_matrix(labels_, outputs_) 
    # confusions = confusions.astype('float') / confusions.sum(axis=1)[:, np.newaxis]

    print('#########\t Confusion Matrix \t#########')
    print(f'\t{title}')
    
    for i in range(len(confusions)):
        row = ''
        for c in confusions[:, i]:
            row += f'{c}\t'
        print(f'^{column[3*i:3*(i+1)]}|\t{row}')

    acc =  balanced_accuracy_score(labels_, outputs_)
    print(f'Balanced Accuracy: {acc}')

    print(f'Classification Report:\n{classification_report(labels_, outputs_)}')
