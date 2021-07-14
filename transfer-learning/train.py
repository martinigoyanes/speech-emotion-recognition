import torch
import time
from utils import *
import numpy as np
import random as rn
from torch import nn
from torch.optim import *

# Los parámetros de train_model son la red (model), el criterio (la loss),
# el optimizador, una estrategia de lr, y las épocas de entrenamiento
def train_model(
    model,
    seed,
    name,
    data,
    device,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    results,
    num_epochs=25,
    patience=8,
    delta=0.0001,
):
    # Fix seeds
    if seed:
        rn.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience, delta=delta, name=f"{name}.pt", verbose=True
    )
    data_size = {"train": len(data["train"]), "val": len(data["val"])}

    since = time.time()
    # Bucle de épocas de entrenamiento
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        train_epoch(
            dataloader,
            device,
            optimizer,
            early_stopping,
            model,
            criterion,
            scheduler,
            data_size,
            results,
        )

        if early_stopping.early_stop:
            print("Early stopped")
            break

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s \nBest CCC: {:.4f}".format(
            time_elapsed // 60, time_elapsed % 60, early_stopping.best_score
        )
    )


def train_epoch(
    dataloader,
    device,
    optimizer,
    early_stopping,
    model,
    criterion,
    scheduler,
    data_size,
    results,
):
    # Cada época tiene entrenamiento y validación
    for phase in ["train", "val"]:
        (
            running_loss,
            running_mae,
            running_mse,
            running_r2,
            running_ccc,
            running_v_ccc,
            running_a_ccc,
            running_d_ccc,
        ) = train_phase(
            phase, model, dataloader, device, optimizer, scheduler, criterion
        )
        # Loss acumulada en la época
        epoch_loss = running_loss / data_size[phase]
        epoch_mae = running_mae / data_size[phase]
        epoch_mse = running_mse / data_size[phase]
        epoch_r2 = running_r2 / data_size[phase]
        epoch_ccc = running_ccc / data_size[phase] / 3
        epoch_v_ccc = running_v_ccc / data_size[phase]
        epoch_a_ccc = running_a_ccc / data_size[phase]
        epoch_d_ccc = running_d_ccc / data_size[phase]

        print(
            "{} R2: {:.4f} MAE: {:.4f} MSE: {:.4f} loss: {:.4f}\tCCC: {:.4f}  v_ccc: {:.4f}  a_ccc: {:.4f}  d_ccc: {:.4f} ".format(
                phase,
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
        # early_stopping needs the validation EPOCH AUC to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        if phase == "val":
            early_stopping(epoch_ccc, model)

        results[phase]["v_ccc"].append(epoch_v_ccc)
        results[phase]["a_ccc"].append(epoch_a_ccc)
        results[phase]["d_ccc"].append(epoch_d_ccc)
        results[phase]["ccc"].append(epoch_ccc)
        results[phase]["loss"].append(epoch_loss)
        results[phase]["mse"].append(epoch_mse)
        results[phase]["r2"].append(epoch_r2)
        results[phase]["mae"].append(epoch_mae)


def train_phase(phase, model, dataloader, device, optimizer, scheduler, criterion):
    running_mae = (
        running_r2
    ) = (
        running_mse
    ) = running_loss = running_ccc = running_v_ccc = running_a_ccc = running_d_ccc = 0.0

    if phase == "train":
        model.train()  # Ponemos el modelo en modo entrenamiento
    else:
        model.eval()  # Ponemos el modelo en modo evaluación

    model.init_hidden()
    # Creamos las variables que almacenarán las salidas y las etiquetas running_loss = running_v_loss = running_a_loss = running_d_loss = running_ccc = running_v_ccc = running_a_ccc = running_d_ccc = 0.0
    # Iteramos sobre los datos.
    for sample in dataloader[phase]:
        audio_input = sample["audio"].to(device)
        text_input = sample["text"].to(device)
        labels = sample["labels"].to(device)
        # print(phase, 'audio_input:', audio_input.size(), 'text_input:', text_input.size())

        # Tamaño del batch
        batchSize = labels.size(0)

        # Ponemos a cero los gradientes
        optimizer.zero_grad()

        # Paso forward
        # registramos operaciones solo en train
        # with torch.autograd.detect_anomaly():
        with torch.set_grad_enabled(phase == "train"):
            if phase == "train":
                model.repackage_hidden()
            outputs = model(audio_input, text_input)
            if phase == "val":
                model.repackage_hidden()

            loss, ccc, v_ccc, a_ccc, d_ccc = criterion(outputs, labels)
            mse = nn.MSELoss()(outputs, labels)
            mae = nn.L1Loss()(outputs, labels)
            loss_r2 = r2_loss(outputs, labels)
            r2 = 1 - loss_r2

            # backward y optimización solo en training
            if phase == "train":
                loss.backward()
                optimizer.step()
                # helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=2.0, norm_type=2
                )
                for p in model.parameters():
                    p.data.add_(p.grad, alpha=-0.001)

        # Sacamos estadísticas y actualizamos variables
        running_loss += loss.item() * batchSize
        running_mse += mse.item() * batchSize
        running_mae += mae.item() * batchSize
        running_r2 += r2.item() * batchSize
        running_ccc += ccc.item() * batchSize
        running_v_ccc += v_ccc.item() * batchSize
        running_a_ccc += a_ccc.item() * batchSize
        running_d_ccc += d_ccc.item() * batchSize

    if phase == "train":
        scheduler.step()

    return (
        running_loss,
        running_mae,
        running_mse,
        running_r2,
        running_ccc,
        running_v_ccc,
        running_a_ccc,
        running_d_ccc,
    )
