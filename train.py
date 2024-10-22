import torch
from model import pde
import numpy as np
import time


def train(
    nb_itt,
    train_loss,
    test_loss,
    poids,
    inputs_train_data,
    outputs_train_data,
    points_pde,
    model,
    loss,
    optimizer,
    X_test_pde,
    X_test_data,
    U_test_data,
    Re,
    time_start,
    f,
    x_std,
    y_std,
    u_mean,
    v_mean,
    p_std,
    t_std,
    u_std,
    v_std,
    batch_size,
):

    nb_it_tot = nb_itt + len(train_loss["total"])
    # Nos datas initiales
    X_train_data = inputs_train_data
    U_train_data = outputs_train_data
    X_train_pde = points_pde
    print(
        f"--------------------------\nStarting at epoch: {len(train_loss['total'])}"
        + "\n--------------------------"
    )
    print(
        f"--------------------------\nStarting at epoch: {len(train_loss['total'])}\n------------"
        + "--------------",
        file=f,
    )
    for epoch in range(len(train_loss["total"]), nb_it_tot):
        indices_batch = torch.randint(
            X_train_pde.size(0), (batch_size * (len(X_train_pde) // batch_size),)
        )
        loss_batch_train = {"total": [], "data": [], "pde": []}
        loss_batch_test = {"total": [], "data": [], "pde": []}
        for batch in range(len(X_train_pde) // batch_size):
            model.train()  # on dit qu'on va entrainer (on a le dropout)
            ## loss du pde
            X_pde_batch = X_train_pde[
                indices_batch[batch * batch_size : (batch + 1) * batch_size], :
            ]
            pred_pde = model(X_pde_batch)
            pred_pde1, pred_pde2, pred_pde3 = pde(
                pred_pde,
                X_pde_batch,
                Re=Re,
                x_std=x_std,
                y_std=y_std,
                u_mean=u_mean,
                v_mean=v_mean,
                p_std=p_std,
                t_std=t_std,
                u_std=u_std,
                v_std=v_std,
            )
            loss_pde = (
                torch.mean(pred_pde1**2)
                + torch.mean(pred_pde2**2)
                + torch.mean(pred_pde3**2)
            )

            # loss des points de data
            pred_data = model(X_train_data)
            loss_data = loss(U_train_data, pred_data)  # (MSE)

            # loss totale
            loss_totale = poids[0] * loss_data + poids[1] * loss_pde
            with torch.no_grad():
                loss_batch_train["total"].append(loss_totale.item())
                loss_batch_train["data"].append(loss_data)
                loss_batch_train["pde"].append(loss_pde)

            # Backpropagation
            loss_totale.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

            # Pour le test :
            model.eval()

            # loss du pde
            test_pde = model(X_test_pde)
            test_pde1, test_pde2, test_pde3 = pde(
                test_pde,
                X_test_pde,
                Re=Re,
                x_std=x_std,
                y_std=y_std,
                u_mean=u_mean,
                v_mean=v_mean,
                p_std=p_std,
                t_std=t_std,
                u_std=u_std,
                v_std=v_std,
            )
            loss_test_pde = (
                torch.mean(test_pde1**2)
                + torch.mean(test_pde2**2)
                + torch.mean(test_pde3**2)
            )  # (MSE)

            # loss de la data
            test_data = model(X_test_data)
            loss_test_data = loss(U_test_data, test_data)  # (MSE)

            # loss totale
            loss_test = poids[0] * loss_test_data + poids[1] * loss_test_pde
            with torch.no_grad():
                loss_batch_test["total"].append(loss_test.item())
                loss_batch_test["data"].append(loss_test_data)
                loss_batch_test["pde"].append(loss_test_pde)
        with torch.no_grad():
            train_loss["total"].append(np.mean(loss_batch_train["total"]))
            train_loss["data"].append(np.mean(loss_batch_train["data"]))
            train_loss["pde"].append(np.mean(loss_batch_train["pde"]))
            test_loss["total"].append(np.mean(loss_batch_test["total"]))
            test_loss["data"].append(np.mean(loss_batch_test["data"]))
            test_loss["pde"].append(np.mean(loss_batch_test["pde"]))
        print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :")
        print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :", file=f)
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}, data: {train_loss['data'][-1]:.3e}, pde: {train_loss['pde'][-1]:.3e}"
        )
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}, data: {train_loss['data'][-1]:.3e}, pde: {train_loss['pde'][-1]:.3e}",
            file=f,
        )
        print(
            f"Test  : loss: {test_loss['total'][-1]:.3e}, data: {test_loss['data'][-1]:.3e}, pde: {test_loss['pde'][-1]:.3e}"
        )
        print(
            f"Test  : loss: {test_loss['total'][-1]:.3e}, data: {test_loss['data'][-1]:.3e}, pde: {test_loss['pde'][-1]:.3e}",
            file=f,
        )

        print(f"time: {time.time()-time_start:.0f}s")
        print(f"time: {time.time()-time_start:.0f}s", file=f)
