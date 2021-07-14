import torch
import time
from utils import *
import numpy as np
import random as rn
from torch import nn
from torch.optim import *
from sklearn.metrics import accuracy_score

# Los parámetros de train_model son la red (model), el criterio (la loss),
# el optimizador, una estrategia de lr, y las épocas de entrenamiento
def train_model(
    model,
    seed,
    name,
    data,
    device,
    dataloader,
    crit_cat,
    crit_dim,
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
            crit_cat,
            crit_dim,
            scheduler,
            data_size,
            results,
        )

        if early_stopping.early_stop:
            print("Early stopped")
            break

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )


def train_epoch(
    dataloader,
    device,
    optimizer,
    early_stopping,
    model,
    crit_cat,
    crit_dim,
    scheduler,
    data_size,
    results,
):
    # Cada época tiene entrenamiento y validación
    for phase in ["train", "val"]:
        (
            running_tot_loss,
            running_cat_loss,
            running_dim_loss,
            running_acc,
            running_ccc,
            running_v_ccc,
            running_a_ccc,
            running_d_ccc,
        ) = train_phase(
            phase, model, dataloader, device, optimizer, scheduler, crit_cat, crit_dim
        )
        
        # Loss acumulada en la época
        epoch_tot_loss = running_tot_loss / data_size[phase]
        epoch_cat_loss = running_cat_loss / data_size[phase]
        epoch_dim_loss = running_dim_loss / data_size[phase]
        epoch_acc = running_acc / data_size[phase]
        epoch_ccc = running_ccc / data_size[phase] / 3
        epoch_v_ccc = running_v_ccc / data_size[phase]
        epoch_a_ccc = running_a_ccc / data_size[phase]
        epoch_d_ccc = running_d_ccc / data_size[phase]
        
        if phase == 'val':
            phase = 'valid' # so that print output is aligned
        print(
            "{} TOT-LOSS: {:.4f} Cat-loss: {:.4f} Dim-loss: {:.4f} \tACC: {:.4f}  CCC: {:.4f}  v_ccc: {:.4f}  a_ccc: {:.4f}  d_ccc: {:.4f} ".format(
                phase,
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
        if phase == 'valid':
            phase = 'val'
        # early_stopping needs the validation EPOCH AUC to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        if phase == "val":
            early_stopping(epoch_acc, model)

        results[phase]["v_ccc"].append(epoch_v_ccc)
        results[phase]["a_ccc"].append(epoch_a_ccc)
        results[phase]["d_ccc"].append(epoch_d_ccc)
        results[phase]["acc"].append(epoch_acc)
        results[phase]["ccc"].append(epoch_ccc)
        results[phase]["cat_loss"].append(epoch_cat_loss)
        results[phase]["dim_loss"].append(epoch_dim_loss)
        results[phase]["tot_loss"].append(epoch_tot_loss)


def train_phase(phase, model, dataloader, device, optimizer, scheduler, crit_cat, crit_dim):
    running_tot_loss = running_cat_loss = running_dim_loss = \
         running_acc = running_ccc = running_v_ccc = running_a_ccc = running_d_ccc = 0.0
    outputs_ = []
    labels_ = []
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
        labels_cat = sample["labels_cat"].to(device)
        labels_dim = sample["labels_dim"].to(device)
        # print(phase, 'audio_input:', audio_input.size(), 'text_input:', text_input.size())

        # Tamaño del batch
        batchSize = labels_cat.size(0)

        # Ponemos a cero los gradientes
        optimizer.zero_grad()

        # Paso forward
        # registramos operaciones solo en train
        with torch.set_grad_enabled(phase == "train"):
            if phase == "train":
                model.repackage_hidden()
            out_cat, out_dim = model(audio_input, text_input)
            if phase == "val":
                model.repackage_hidden()

            dim_loss, ccc, v_ccc, a_ccc, d_ccc = crit_dim(
                out_dim, labels_dim
            )
            cat_loss = crit_cat(out_cat, labels_cat)

            tot_loss = 0.7*cat_loss + 0.3*dim_loss
            # tot_loss = cat_loss + dim_loss
            # backward y optimización solo en training
            if phase == "train":
                tot_loss.backward()
                optimizer.step()
                # helps prevent the exploding gradient problem in RNNs / LSTMs.
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(), max_norm=2.0, norm_type=2
                # )
                # for p in model.parameters():
                #     p.data.add_(p.grad, alpha=-0.001)
            out_cat=nn.functional.softmax(out_cat.data,dim=1)
            indices, preds = torch.max(out_cat, dim=1)
            for i in range(len(preds)):
                outputs_.append(preds[i].item())
                labels_.append(labels_cat[i].item())
            acc =  accuracy_score(labels_, outputs_)

         # Sacamos estadísticas y actualizamos variables
        running_tot_loss += tot_loss.item() * batchSize
        running_cat_loss += cat_loss.item() * batchSize
        running_dim_loss += dim_loss.item() * batchSize
        running_acc += acc.item() * batchSize
        running_ccc += ccc.item() * batchSize
        running_v_ccc += v_ccc.item() * batchSize
        running_a_ccc += a_ccc.item() * batchSize
        running_d_ccc += d_ccc.item() * batchSize

    if phase == "train":
        scheduler.step()

    return (
        running_tot_loss,
        running_cat_loss,
        running_dim_loss,
        running_acc,
        running_ccc,
        running_v_ccc,
        running_a_ccc,
        running_d_ccc,
    )