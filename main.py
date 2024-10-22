# Avec Z et le choix des points avec une certaine proba
from deepxrte.geometry import Rectangle
import torch
import torch.nn as nn
import torch.optim as optim
from model import PINNs
from utils import read_csv, write_csv
from train import train
from pathlib import Path
import time
import numpy as np
import scipy.io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

time_start = time.time()

############# LES VARIABLES ################

folder_result = "2_initialized_xavier"  # le nom du dossier de résultat

random_seed_train = None  # la seed de train, la changer pour avoir de nouveau résultats
random_seed_test = 2002  # la seed de test, la même pour pouvoir comparer


##### Le modèle de résolution de l'équation de la chaleur
nb_itt = 5  # le nb d'epoch
batch_size = 128
poids = [1, 1]  # les poids pour la loss

n_pde_train = 8000


n_data_test = 5000
n_pde_test = 5000

Re = 3900

lr = 1e-3


##### Le code ###############################
###############################################

# La data
mat_data = scipy.io.loadmat("cylinder_data.mat")
data = mat_data["stack"]

mat_data_reduce = scipy.io.loadmat("cylinder_data_25.mat")
data_reduce = mat_data_reduce["stack"]

# On adimensionne la data
x, y, t = data[:, 0], data[:, 1], data[:, 2]
x, y = x - x.min(), y - y.min()
u, v, p = data[:, 3], data[:, 4], data[:, 5]

x_norm = (x - x.mean()) / x.std()
y_norm = (y - y.mean()) / y.std()
t_norm = (t - t.mean()) / t.std()
p_norm = (p - p.mean()) / p.std()
u_norm = (u - u.mean()) / u.std()
v_norm = (v - v.mean()) / v.std()


X = np.array([x_norm, y_norm, t_norm], dtype=np.float32).T
U = np.array([u_norm, v_norm, p_norm], dtype=np.float32).T

x_reduce, y_reduce, t_reduce = data_reduce[:, 0], data_reduce[:, 1], data_reduce[:, 2]
u_reduce, v_reduce, p_reduce = data_reduce[:, 3], data_reduce[:, 4], data_reduce[:, 5]

x_norm_reduce = (x_reduce - x_reduce.mean()) / x_reduce.std()
y_norm_reduce = (y_reduce - y_reduce.mean()) / y_reduce.std()
t_norm_reduce = (t_reduce - t_reduce.mean()) / t_reduce.std()
p_norm_reduce = (p_reduce - p_reduce.mean()) / p_reduce.std()
u_norm_reduce = (u_reduce - u_reduce.mean()) / u_reduce.std()
v_norm_reduce = (v_reduce - v_reduce.mean()) / v_reduce.std()


X_reduce = np.array([x_norm_reduce, y_norm_reduce, t_norm_reduce], dtype=np.float32).T
U_reduce = np.array([u_norm_reduce, v_norm_reduce, p_norm_reduce], dtype=np.float32).T

t_norm_reduce_min = t_norm_reduce.min()
t_norm_reduce_max = t_norm_reduce.max()
t_max_reduce = t_reduce.max()

x_norm_reduce_max = x_norm_reduce.max()
y_norm_reduce_max = y_norm_reduce.max()


# On regarde si le dossier existe
dossier = Path(folder_result)
dossier.mkdir(parents=True, exist_ok=True)

if random_seed_train is not None:
    torch.manual_seed(random_seed_train)


rectangle = Rectangle(
    x_max=x_norm.max(), y_max=y_norm.max(), t_min=t_norm.min(), t_max=t_norm.max()
)  # le domaine de résolution


# les points initiaux du train
# Les points de pde


### Pour train
# les points pour la pde
points_pde = rectangle.generate_random(n_pde_train).to(device)


inputs_train = torch.from_numpy(X_reduce).requires_grad_().to(device)
outputs_train = torch.from_numpy(U_reduce).requires_grad_().to(device)


### Pour test
torch.manual_seed(random_seed_test)
np.random.seed(random_seed_test)
X_test_pde = rectangle.generate_random(n_pde_test).to(device)
points_coloc_test = np.random.choice(len(X), n_data_test, replace=False)
X_test_data = torch.from_numpy(X[points_coloc_test]).requires_grad_().to(device)
U_test_data = torch.from_numpy(U[points_coloc_test]).requires_grad_().to(device)


# Initialiser le modèle
model = PINNs().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss = nn.MSELoss()

# On plot les print dans un fichier texte
with open(folder_result + "/print.txt", "a") as f:
    # On regarde si notre modèle n'existe pas déjà
    if Path(folder_result + "/model_weights.pth").exists():
        # Charger l'état du modèle et de l'optimiseur
        checkpoint = torch.load(folder_result + "/model_weights.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("\nModèle chargé\n", file=f)
        print("\nModèle chargé\n")
        csv_train = read_csv(folder_result + "/train_loss.csv")
        csv_test = read_csv(folder_result + "/test_loss.csv")
        train_loss = {
            "total": list(csv_train["total"]),
            "data": list(csv_train["data"]),
            "pde": list(csv_train["pde"]),
        }
        test_loss = {
            "total": list(csv_test["total"]),
            "data": list(csv_test["data"]),
            "pde": list(csv_test["pde"]),
        }
        print("\nLoss chargée\n", file=f)
        print("\nLoss chargée\n")

    else:
        print("Nouveau modèle\n", file=f)
        print("Nouveau modèle\n")
        train_loss = {"total": [], "data": [], "pde": []}
        test_loss = {"total": [], "data": [], "pde": []}

    ######## On entraine le modèle
    ###############################################
    train(
        nb_itt=nb_itt,
        train_loss=train_loss,
        test_loss=test_loss,
        poids=poids,
        inputs_train_data=inputs_train,
        outputs_train_data=outputs_train,
        points_pde=points_pde,
        model=model,
        loss=loss,
        optimizer=optimizer,
        X_test_pde=X_test_pde,
        X_test_data=X_test_data,
        U_test_data=U_test_data,
        Re=Re,
        time_start=time_start,
        f=f,
        u_mean=u.mean(),
        v_mean=v.mean(),
        x_std=x.std(),
        y_std=y.std(),
        t_std=t.std(),
        u_std=u.std(),
        v_std=v.std(),
        p_std=p.std(),
        batch_size=batch_size,
    )

####### On save le model et les losses

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    folder_result + "/model_weights.pth",
)
write_csv(train_loss, folder_result, file_name="/train_loss.csv")
write_csv(test_loss, folder_result, file_name="/test_loss.csv")


dossier_end = Path(folder_result + f"/epoch{len(train_loss['total'])}")
dossier_end.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    folder_result + f"/epoch{len(train_loss['total'])}" + "/model_weights.pth",
)
write_csv(
    train_loss,
    folder_result + f"/epoch{len(train_loss['total'])}",
    file_name="/train_loss.csv",
)
write_csv(
    test_loss,
    folder_result + f"/epoch{len(train_loss['total'])}",
    file_name="/test_loss.csv",
)
