from train import *
from utils import *
from Network import *
from torch import optim
import torch
import random as rn
import numpy as np
import time
from SpeechDataset import *

"""
Function to perform experiments with the model and return the average results. The model is trained
and tested multiple times to see if the experimental model gives better or worse results 
than the baseline model.
"""


def experiment(
    learning,
    params,
    data,
    device,
    path,
    crit_cat,
    crit_dim,
    num_runs,
):
    experiment_results = []
    since = time.time()

    for i in range(num_runs):
        model, optimizer_ft, results, scheduler, dataset, dataloader = clear_state(
            learning, data, params, path, device
        )
        if i == 0:
            pretty_print(
                dataset["all"], dataset["train"], dataset["val"], dataset["test"]
            )
            print(model)

        print(f"#################\tRUN {i}\t#################")

        train_model(
            model,
            learning["seed"],
            params["name"],
            dataset,
            device,
            dataloader,
            crit_cat,
            crit_dim,
            optimizer_ft,
            scheduler,
            results,
            learning["epochs"],
            learning["patience"],
            learning["delta"],
        )
        model.load_state_dict(torch.load(f'{params["name"]}.pt'))
        tot_loss, cat_loss, dim_loss, acc, ccc, v_ccc, a_ccc, d_ccc = test_model(
            model, crit_cat, crit_dim, dataloader["test"], dataset["test"], device
        )
        results["test"] = {
            "tot_loss": tot_loss,
            "cat_loss": cat_loss,
            "dim_loss": dim_loss,
            "acc": acc,
            "ccc": ccc,
            "v_ccc": v_ccc,
            "a_ccc": a_ccc,
            "d_ccc": d_ccc,
        }
        experiment_results.append(results)

    # visualize experiment
    print("-" * 50)
    print("\t\t\tRESULTS")
    print("-" * 50)

    for phase in ["train", "val"]:
        acc_list = torch.FloatTensor(
            [torch.mean(torch.FloatTensor(r[phase]["acc"])) for r in experiment_results]
        )
        ccc_list = torch.FloatTensor(
            [torch.mean(torch.FloatTensor(r[phase]["ccc"])) for r in experiment_results]
        )
        tot_loss_list = torch.FloatTensor(
            [
                torch.mean(torch.FloatTensor(r[phase]["tot_loss"]))
                for r in experiment_results
            ]
        )
        cat_loss_list = torch.FloatTensor(
            [
                torch.mean(torch.FloatTensor(r[phase]["cat_loss"]))
                for r in experiment_results
            ]
        )
        dim_loss_list = torch.FloatTensor(
            [
                torch.mean(torch.FloatTensor(r[phase]["dim_loss"]))
                for r in experiment_results
            ]
        )
        v_list = torch.FloatTensor(
            [
                torch.mean(torch.FloatTensor(r[phase]["v_ccc"]))
                for r in experiment_results
            ]
        )
        a_list = torch.FloatTensor(
            [
                torch.mean(torch.FloatTensor(r[phase]["a_ccc"]))
                for r in experiment_results
            ]
        )
        d_list = torch.FloatTensor(
            [
                torch.mean(torch.FloatTensor(r[phase]["d_ccc"]))
                for r in experiment_results
            ]
        )
        title = "Train" if phase == "train" else "Validation"
        print(f"#################\t{title}\t#################")
        print(
            "\tTOT_LOSS: {:.4f}+-{:.4f}\n\tCat-loss: {:.4f}+-{:.4f}\n\tDim-loss: {:.4f}+-{:.4f}\n\tACC: {:.4f}+-{:.4f} \n\tCCC: {:.4f}+-{:.4f} \n\tv_ccc: {:.4f}+-{:.4f} \n\ta_ccc: {:.4f}+-{:.4f}  \n\td_ccc: {:.4f}+-{:.4f}".format(
                torch.mean(tot_loss_list),
                torch.std(tot_loss_list),
                torch.mean(cat_loss_list),
                torch.std(cat_loss_list),
                torch.mean(dim_loss_list),
                torch.std(dim_loss_list),
                torch.mean(acc_list),
                torch.std(acc_list),
                torch.mean(ccc_list),
                torch.std(ccc_list),
                torch.mean(v_list),
                torch.std(v_list),
                torch.mean(a_list),
                torch.std(a_list),
                torch.mean(d_list),
                torch.std(d_list),
            )
        )

    tot_loss_list = torch.FloatTensor(
        [r["test"]["tot_loss"] for r in experiment_results]
    )
    cat_loss_list = torch.FloatTensor(
        [r["test"]["cat_loss"] for r in experiment_results]
    )
    dim_loss_list = torch.FloatTensor(
        [r["test"]["dim_loss"] for r in experiment_results]
    )
    acc_list = torch.FloatTensor([r["test"]["acc"] for r in experiment_results])
    ccc_list = torch.FloatTensor([r["test"]["ccc"] for r in experiment_results])
    v_list = torch.FloatTensor([r["test"]["v_ccc"] for r in experiment_results])
    a_list = torch.FloatTensor([r["test"]["a_ccc"] for r in experiment_results])
    d_list = torch.FloatTensor([r["test"]["d_ccc"] for r in experiment_results])
    print(f"#################\tTest\t#################")
    print(
        "\tTOT_LOSS: {:.4f}+-{:.4f}\n\tCat-loss: {:.4f}+-{:.4f}\n\tDim-loss: {:.4f}+-{:.4f}\n\tACC: {:.4f}+-{:.4f} \n\tCCC: {:.4f}+-{:.4f} \n\tv_ccc: {:.4f}+-{:.4f} \n\ta_ccc: {:.4f}+-{:.4f}  \n\td_ccc: {:.4f}+-{:.4f}".format(
            torch.mean(tot_loss_list),
            torch.std(tot_loss_list),
            torch.mean(cat_loss_list),
            torch.std(cat_loss_list),
            torch.mean(dim_loss_list),
            torch.std(dim_loss_list),
            torch.mean(acc_list),
            torch.std(acc_list),
            torch.mean(ccc_list),
            torch.std(ccc_list),
            torch.mean(v_list),
            torch.std(v_list),
            torch.mean(a_list),
            torch.std(a_list),
            torch.mean(d_list),
            torch.std(d_list),
        )
    )
    best_dim = torch.argmax(ccc_list)
    best_cat = torch.argmax(acc_list)
    print(
        f"Best model in testing for REGRESSION: Model {best_dim} \t CCC = {ccc_list[best_dim]:.4f}\n"
        f"Best model in testing for CLASSIFICATION: Model {best_cat} \t ACC = {acc_list[best_cat]:.4f}"
    )
    plot_training(experiment_results[best_cat], params["name"])

    time_elapsed = time.time() - since
    print("-" * 50)
    print(
        "Experiment completed in {:.0f}m {:.0f}s \n".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )


def clear_state(
    learning,
    data,
    params,
    path,
    device,
):
    from torch import nn

    dataset, dataloader = load_data(
        data["classes"],
        data["audio_feat"],
        data["text_feat"],
        data["labels_cat"],
        data["labels_dim"],
        path,
        params['text_net']['timesteps'],
        params['audio_net']['timesteps'],
        data["ratio"]["train"],
        data["ratio"]["val"],
        learning["train_bsz"],
        learning["test_bsz"],
    )

    model = create_network(
        learning["train_bsz"],
        data["vocab_size"],
        data["embeddings"],
        params["audio_net"],
        params["text_net"],
        params["net"],
        len(dataset["all"].classes),
        device,
    )
    # Usaremos Adam para optimizar
    optimizer_ft = optim.RMSprop(model.parameters(), lr=learning["lr"])
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft,
        step_size=learning["scheduler"]["step"],
        gamma=learning["scheduler"]["gamma"],
    )

    # most_common, _ = torch.max(torch.FloatTensor([v[1] for emo, v in dataset['train'].imbalances.items()]), dim=0)
    # weights = torch.FloatTensor([most_common/v[1] for emo, v in dataset['train'].imbalances.items()]).to(device)
    # print(f'Weights:\t{weights}')
    # crit_cat = nn.CrossEntropyLoss(weight=weights)

    results = {
        "train": {
            "v_ccc": [],
            "a_ccc": [],
            "d_ccc": [],
            "acc": [],
            "ccc": [],
            "cat_loss": [],
            "dim_loss": [],
            "tot_loss": [],
        },
        "val": {
            "v_ccc": [],
            "a_ccc": [],
            "d_ccc": [],
            "acc": [],
            "ccc": [],
            "cat_loss": [],
            "dim_loss": [],
            "tot_loss": [],
        },
        "test": {},
    }

    return model, optimizer_ft, results, exp_lr_scheduler, dataset, dataloader#, crit_cat
