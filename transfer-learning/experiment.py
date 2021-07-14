from train import *
from utils import *
from Network import *
from torch import optim
import torch
import random as rn
import numpy as np
import time
import copy
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
    criterion,
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

        if params["pre_training"] == False:
            ### Load the whole thing
            # model.load_state_dict(torch.load(f"best-{params['name']}.pt"))

            ### Load embeddings
            # loaded_params = torch.load(f"best-{params['name']}.pt")
            # embeds = loaded_params['text_net.embedding.weight']
            # model.text_net.embedding.weight = nn.Parameter(embeds)

            ### Load audio net
            loaded_params = torch.load(f"best-{params['name']}.pt")
            audio_params = ['audio_net.batch_norm.weight', 'audio_net.batch_norm.bias', 
            'audio_net.batch_norm.running_mean', 'audio_net.batch_norm.running_var', 
            'audio_net.layer1.weight_ih_l0','audio_net.layer1.weight_hh_l0', 
            'audio_net.layer1.bias_ih_l0', 'audio_net.layer1.bias_hh_l0', 
            'audio_net.layer2.weight_ih_l0', 'audio_net.layer2.weight_hh_l0', 
            'audio_net.layer2.bias_ih_l0', 
            'audio_net.layer2.bias_hh_l0', 'audio_net.layer3.weight_ih_l0', 'audio_net.layer3.weight_hh_l0', 
            'audio_net.layer3.bias_ih_l0', 'audio_net.layer3.bias_hh_l0']
            
            for p_name in audio_params:
                p_value = loaded_params[p_name]
                model_attr = copy.deepcopy(model)
                attr_list = p_name.split('.')
                for i in range(len(attr_list)-1):
                    model_attr = getattr(model_attr, attr_list[i])

                setattr(model_attr, attr_list[i+1], nn.Parameter(p_value))

            ### Load the dense layers
            loaded_params = torch.load(f"best-{params['name']}.pt")
            dense_params = ['linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias']
            for p_name in dense_params:
                p_value = loaded_params[p_name]
                model_attr = copy.deepcopy(model)
                attr_list = p_name.split('.')
                for i in range(len(attr_list)-1):
                    model_attr = getattr(model_attr, attr_list[i])

                setattr(model_attr, attr_list[i+1], nn.Parameter(p_value))

            ### Load the whole Linguistic Network
            loaded_params = torch.load(f"best-{params['name']}.pt")
            text_params = ['text_net.layer1.weight_ih_l0', 'text_net.layer1.weight_hh_l0', 'text_net.layer1.bias_ih_l0', 
            'text_net.layer1.bias_hh_l0', 'text_net.layer2.weight_ih_l0', 'text_net.layer2.weight_hh_l0', 
            'text_net.layer2.bias_ih_l0', 'text_net.layer2.bias_hh_l0', 'text_net.layer3.weight_ih_l0', 
            'text_net.layer3.weight_hh_l0', 'text_net.layer3.bias_ih_l0', 'text_net.layer3.bias_hh_l0', 
            'text_net.linear.weight', 'text_net.linear.bias', 'text_net.embedding.weight']
            for p_name in text_params:
                p_value = loaded_params[p_name]
                model_attr = copy.deepcopy(model)
                attr_list = p_name.split('.')
                for i in range(len(attr_list)-1):
                    model_attr = getattr(model_attr, attr_list[i])

                setattr(model_attr, attr_list[i+1], nn.Parameter(p_value))

        train_model(
            model,
            learning["seed"],
            params["name"],
            dataset,
            device,
            dataloader,
            criterion,
            optimizer_ft,
            scheduler,
            results,
            learning["epochs"],
            learning["patience"],
            learning["delta"],
        )
        model.load_state_dict(torch.load(f'{params["name"]}.pt'))
        
        (mse,
        mae,
        r2,
        ccc,
        v_ccc,
        a_ccc,
        d_ccc) = test_model(
            model, criterion, dataloader["test"], dataset["test"], device
        )

        results["test"] = {"mse":mse, "mae":mae, "r2": r2, "ccc": ccc, "v_ccc": v_ccc, "a_ccc": a_ccc, "d_ccc": d_ccc}
        results["model_state"] = model.state_dict()
        experiment_results.append(results)

    # visualize experiment
    print("-" * 50)
    print("\t\t\tRESULTS")
    print("-" * 50)

    for phase in ["train", "val"]:
        mse_list = torch.FloatTensor(
            [torch.mean(torch.FloatTensor(r[phase]["mse"])) for r in experiment_results]
        )
        mae_list = torch.FloatTensor(
            [torch.mean(torch.FloatTensor(r[phase]["mae"])) for r in experiment_results]
        )
        r2_list = torch.FloatTensor(
            [torch.mean(torch.FloatTensor(r[phase]["r2"])) for r in experiment_results]
        )
        ccc_list = torch.FloatTensor(
            [torch.mean(torch.FloatTensor(r[phase]["ccc"])) for r in experiment_results]
        )
        loss_list = torch.FloatTensor(
            [
                torch.mean(torch.FloatTensor(r[phase]["loss"]))
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
            "\tR2: {:.4f}+-{:.4f} \n\t MAE: {:.4f}+-{:.4f} MSE: {:.4f}+-{:.4f} \n\tloss: {:.4f}+-{:.4f} \n\tCCC: {:.4f}+-{:.4f} \n\tv_ccc: {:.4f}+-{:.4f} \n\ta_ccc: {:.4f}+-{:.4f}  \n\td_ccc: {:.4f}+-{:.4f}".format(
                torch.mean(r2_list),
                torch.std(r2_list),
                torch.mean(mae_list),
                torch.std(mae_list),
                torch.mean(mse_list),
                torch.std(mse_list),
                torch.mean(loss_list),
                torch.std(loss_list),
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

    r2_list = torch.FloatTensor([r["test"]["r2"] for r in experiment_results])
    mae_list = torch.FloatTensor([r["test"]["mae"] for r in experiment_results])
    mse_list = torch.FloatTensor([r["test"]["mse"] for r in experiment_results])
    ccc_list = torch.FloatTensor([r["test"]["ccc"] for r in experiment_results])
    v_list = torch.FloatTensor([r["test"]["v_ccc"] for r in experiment_results])
    a_list = torch.FloatTensor([r["test"]["a_ccc"] for r in experiment_results])
    d_list = torch.FloatTensor([r["test"]["d_ccc"] for r in experiment_results])

    print(f"#################\tTest\t#################")
    print(
        "\t R2: {:.4f}+-{:.4f} \n\t MAE: {:.4f}+-{:.4f} \n\t MSE: {:.4f}+-{:.4f} \n\t ccc: {:.4f}+-{:.4f} \n\tv_ccc: {:.4f}+-{:.4f} \n\ta_ccc: {:.4f}+-{:.4f}  \n\td_ccc: {:.4f}+-{:.4f}".format(
            torch.mean(r2_list),
            torch.std(r2_list),
            torch.mean(mae_list),
            torch.std(mae_list),
            torch.mean(mse_list),
            torch.std(mse_list),
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
    best = torch.argmax(ccc_list)
    print(
        f"Best model in testing: Model {best} \t R2 = {r2_list[best]:.4f} MAE = {mae_list[best]:.4f} MSE = {mse_list[best]:.4f} ccc = {ccc_list[best]:.4f}\
                 \t[{v_list[best]:.4f}, {a_list[best]:.4f}, {d_list[best]:.4f}]"
    )
    plot_training(experiment_results[best], params["name"])

    if params["pre_training"] == True:
        print('Saved best model')
        torch.save(experiment_results[best]["model_state"], f"best-{params['name']}.pt")

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
        data["audio_feat"],
        data["text_feat"],
        data["labels"],
        path,
        params["text_net"]["timesteps"],
        params["audio_net"]["timesteps"],
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
        device,
    )
    # Usaremos Adam para optimizar
    optimizer_ft = optim.RMSprop(model.parameters(), lr=learning["lr"])
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft,
        step_size=learning["scheduler"]["step"],
        gamma=learning["scheduler"]["gamma"],
    )
    results = {
        "train": {
            "v_ccc": [],
            "a_ccc": [],
            "d_ccc": [],
            "ccc": [],
            "loss": [],
            "mse": [],
            "mae": [],
            "r2": [],
        },
        "val": {
            "v_ccc": [],
            "a_ccc": [],
            "d_ccc": [],
            "ccc": [],
            "loss": [],
            "mse": [],
            "mae": [],
            "r2": [],
        },
        "test": {},
        "model_state": {}
    }

    return model, optimizer_ft, results, exp_lr_scheduler, dataset, dataloader
