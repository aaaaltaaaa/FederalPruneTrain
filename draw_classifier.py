
import numpy as np
import seaborn as sns

sns.set_theme()

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs

nclass = 3
current_palette = sns.color_palette("muted", n_colors=nclass)
cmap = ListedColormap(sns.color_palette(current_palette).as_hex())

centers = [[0, 0], [-3, 3], [-3, -3], [3, 2], [3, -2]]
classes_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
X, y = make_blobs(n_samples=[1000, 500, 500, 500, 500], centers=centers, cluster_std=0.8,
                  n_features=10, random_state=7)

transformation1 = [[0.5, 0], [0, 1]]
transformation2 = [[0.75, 0], [0, 0.75]]
for i in range(len(X)):
    label = y[i]
    if label == 0 or label == 1:
        X[i] = np.dot(X[i], transformation1)
    else:
        X[i] = np.dot(X[i], transformation2)

# Divide user data
sub_class_num = 5
Xs = [0 for _ in range(sub_class_num)]
for i in range(sub_class_num):
    Xs[i] = X[y == i]

nuser = 3
userX = [0 for _ in range(nuser)]
userY = [0 for _ in range(nuser)]

divides = [[0, 0.3, 0.7, 1], [0, 0.2 ,0.6,1], [0, 0.2, 0, 1], [0 ,0.95 ,0.95,1],[0,0,0.8,0]]

for i in range(nuser):
    tempX, tempY = [], []
    for j in range(sub_class_num):
        l = len(Xs[j])
        start, end = int(l * divides[j][i]), int(l * divides[j][i + 1])
        tempX.append(Xs[j][start:end])
        if j == 0:
            tempY.append(np.array([j for _ in range(end - start)]))
        else:
            tempY.append(np.array([classes_map[j] for _ in range(end - start)]))
    userX[i] = np.concatenate(tempX, axis=0)
    userY[i] = np.concatenate(tempY, axis=0)

for i in range(len(y)):
    label = y[i]
    y[i] = classes_map[y[i]]
for i in range(nuser):
    print(len(userY[i]))


def plot_scatter(X, y, title, path):
    plt.figure(figsize=(6, 6))
    plt.xticks([-4.0, -2.0, 0.0, 2.0, 4.0])
    plt.xlim([-4.0, 4.0])
    plt.ylim([-4.0, 4.0])
    plt.title(title, size=18)
    plt.scatter(X[:, 0], X[:, 1], s=15, c=y, cmap=cmap, alpha=0.5)
    plt.savefig(path)


#plot_scatter(X, y, 'Total data', 'total_data.png')

# for i in range(nuser):
#     plot_scatter(userX[i], userY[i], f"User {i + 1} data", f"./data/plot/user_data_{i + 1}.pdf")

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

ce_loss = nn.CrossEntropyLoss(reduction="mean")


def kl_loss(student_logits, teacher_logits):
    divergence = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    )
    return divergence


class MLP(nn.Module):
    def __init__(self, hidden_layer_sizes=None):
        super(MLP, self).__init__()
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [8, 8, 8]
        self.fc = nn.Sequential(
            nn.Linear(2, hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[2], 3)
        )

    def forward(self, x):
        return self.fc(x)


class MLPClassifier():
    def __init__(self, hidden_layer_sizes=[8, 8, 8], max_iter=100):
        self.mlp = MLP(hidden_layer_sizes)
        self.mlp = self.mlp.cuda()
        self.max_iter = max_iter

    def fit(self, X, y, kl=False, teachers=None, pos=0):

        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.001)
        for _ in range(self.max_iter):
            output = self.mlp(X)
            loss =  ce_loss(output, y)
            if kl == True:
                loss2 = 0
                for i,teacher in enumerate(teachers):
                    if teacher == 0:
                        continue
                    teacher_output = teacher(X)
                    loss2 += 10 * kl_loss(output, teacher_output.detach())
                loss = loss + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            output = self.mlp(X)
        y = torch.argmax(output, dim=1)
        return y

    def predict_proba(self, X):
        with torch.no_grad():
            output = self.mlp(X)
        y = torch.softmax(output, dim=1)
        return y


def plot_decision_boundary(model, X, Y, nclass, title, path,
                           xrg=None, yrg=None, Nx=300, Ny=300,
                           figsize=[6, 6], alpha=0.7):
    try:
        getattr(model, 'predict')
    except:
        print("model do not have method predict 'predict' ")
        return None

    x1 = X[:, 0].min() - 0.1 * (X[:, 0].max() - X[:, 0].min())
    x2 = X[:, 0].max() + 0.1 * (X[:, 0].max() - X[:, 0].min())
    y1 = X[:, 1].min() - 0.1 * (X[:, 1].max() - X[:, 1].min())
    y2 = X[:, 1].max() + 0.1 * (X[:, 1].max() - X[:, 1].min())

    if xrg is None:
        xrg = [x1, x2]
    if yrg is None:
        yrg = [y1, y2]

    # generate grid and mesh
    xgrid = np.arange(xrg[0], xrg[1], 1. * (xrg[1] - xrg[0]) / Nx)
    ygrid = np.arange(yrg[0], yrg[1], 1. * (yrg[1] - yrg[0]) / Ny)
    xx, yy = np.meshgrid(xgrid, ygrid)

    # generate X for model prediction and predict the Yp

    X_full_grid = np.array(list(zip(np.ravel(xx), np.ravel(yy))))
    X_full_grid = torch.tensor(X_full_grid, dtype=torch.float32).cuda()
    #Yp = model.predict(X_full_grid)
    Ypp = model.predict_proba(X_full_grid)
    Ypp = Ypp.cpu()
    print(Ypp)

    # initialize figure & axes object
    plt.figure(figsize=figsize)

    # plot probability surface
    current_palette = sns.color_palette("muted", n_colors=nclass)
    zz = np.dot(Ypp, sns.color_palette(current_palette))
    zz_r = zz.reshape(xx.shape[0], xx.shape[1], 3)
    plt.imshow(zz_r, origin='lower', interpolation=None,
               extent=[xrg[0], xrg[1], yrg[0], yrg[1]],
               alpha=alpha)
    plt.scatter(X[:, 0], X[:, 1], s=15, c=Y, cmap=cmap, alpha=0.5)
    plt.xlim(xrg)
    plt.ylim(yrg)
    plt.xticks(np.arange(xrg[0], xrg[1] + 1, 2), size=20)
    plt.yticks(np.arange(yrg[0], yrg[1] + 1, 1), size=20)
    plt.title(title, size=20)

    plt.savefig(path)

def model_avg(avg_model, clients_models):
    for param in avg_model.mlp.parameters():
        param.data = torch.zeros_like(param.data)
    for i in range(len(clients_models)):
        for server_param, user_param in zip(avg_model.mlp.parameters(), clients_models[i].mlp.parameters()):
            server_param.data = server_param.data + user_param.data.clone() / nuser


    return avg_model


nuser = 3
torch.manual_seed(42)
import copy
import random
for i in range(nuser):
    userX[i], userY[i] = torch.tensor(userX[i],dtype=torch.float32), torch.tensor(userY[i],dtype=torch.int64)
    userX[i], userY[i] = userX[i].cuda(), userY[i].cuda()

avg_model = MLPClassifier([128, 128, 128],max_iter=10)
g_avg_model = copy.deepcopy(avg_model)
uniform_users = [[0,1,2],[0,1,2]]
epoch = 0
buffer = [0 for _ in range(len(uniform_users))]

while epoch < 50:
    #users = total_users

    sample_users = uniform_users

    for pos,users in enumerate(uniform_users):
        clfs, clfs_KD = [], []
        for i in users:
            clf = copy.deepcopy(avg_model)
            clf.mlp = clf.mlp.cuda()
            clf.fit(userX[i], userY[i])
            if (epoch + 1) % 50 == 0:
                plot_decision_boundary(clf, X=userX[i].cpu(), Y=userY[i].cpu(), nclass=3, title=f'User {i+1} without distillation', path=f"./data/plot/user_noKD_bound_{i+1}.pdf", xrg=[-4.0, 4.0], yrg=[-4.0, 4.0])
            clfs.append(clf)
            plt.show()

        if epoch > 0:
            for i in users:
                clf = copy.deepcopy(g_avg_model)
                clf.mlp = clf.mlp.cuda()
                # userX[i], userY[i] = torch.tensor(userX[i],dtype=torch.float32), torch.tensor(userY[i],dtype=torch.int64)
                clf.fit(userX[i], userY[i], kl=True, teachers=buffer,pos = pos)
                if (epoch + 1) % 50 == 0:
                    plot_decision_boundary(clf, X=userX[i].cpu(), Y=userY[i].cpu(), nclass=3, title=f'User {i+1} with global distillation', path=f"./data/plot/user_KD_bound_{i+1}.pdf", xrg=[-4.0, 4.0], yrg=[-4.0, 4.0])
                clfs_KD.append(clf)
                plt.show()
        else:
            clfs_KD = clfs

        avg_model = model_avg(avg_model, clients_models=clfs)
        g_avg_model = model_avg(g_avg_model, clients_models=clfs_KD)

        epoch += 1
        if epoch > 5:
            buffer[pos] = g_avg_model.mlp

        if (epoch) % 50 == 0:
            plot_decision_boundary(avg_model, X=X, Y=y, nclass=3, title=f'Global Model of FedAvg',
                               path=f'./data/plot/avg_{epoch}.pdf', xrg=[-4.0, 4.0],
                               yrg=[-4.0, 4.0])
            plt.show()

            plot_decision_boundary(g_avg_model, X=X, Y=y, nclass=3,
                                   title=f'Global Model with distillation',
                                   path=f'./data/plot/avg_KD_{epoch}.pdf', xrg=[-4.0, 4.0],
                                   yrg=[-4.0, 4.0])
            plt.show()


