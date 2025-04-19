# -*- coding: utf-8 -*-
# @Time    : 2025/4/18 19:45
# @Author  : Ginger
# @FileName: DEMVC.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from dataloader import load_data
from utils import *

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# Cifar10
# Cifar100
# Prokaryotic
# Synthetic3d
Dataname = 'MNIST-USPS'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--pre_epochs", default=200)
parser.add_argument("--kl_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--feature_dim", default=10)
parser.add_argument("--temperature", default=1)
parser.add_argument('--update_interval', default=1, type=int)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "MNIST-USPS":
    args.con_epochs = 50
    seed = 10
if args.dataset == "BDGP":
    args.con_epochs = 10 # 20
    seed = 30
if args.dataset == "CCV":
    args.con_epochs = 50 # 100
    seed = 100
    args.tune_epochs = 200
if args.dataset == "Fashion":
    args.con_epochs = 50 # 100
    seed = 10
if args.dataset == "Caltech-2V":
    args.con_epochs = 100
    seed = 200
    args.tune_epochs = 200
if args.dataset == "Caltech-3V":
    args.con_epochs = 100
    seed = 30
if args.dataset == "Caltech-4V":
    args.con_epochs = 100
    seed = 100
if args.dataset == "Caltech-5V":
    args.con_epochs = 100
    seed = 1000000
if args.dataset == "Cifar10":
    args.con_epochs = 10
    seed = 10
if args.dataset == "Cifar100":
    args.con_epochs = 20
    seed = 10
if args.dataset == "Prokaryotic":
    args.con_epochs = 20
    seed = 10000
if args.dataset == "Synthetic3d":
    args.con_epochs = 100
    seed = 100

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last = True
    )


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs,_,_ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))


def KL_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    kloss = torch.nn.KLDivLoss(reduction='batchmean')

    data = [torch.from_numpy(x).float().to(device) for x in x_all]

    if epoch % args.update_interval == 0:
        with torch.no_grad():  # 禁用梯度计算
             _, _, tmp_qs = model(data)
        p = target_distribution_multi_view(tmp_qs)


    for batch_idx, (xs, _, idx) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        xrs, zs,qs = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(0.1 * kloss(qs[v].log(), p[v][idx]))
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


if __name__ == '__main__':

    accs = []
    nmis = []
    purs = []
    if not os.path.exists('./models'):
        os.makedirs('./models')

    x_all = extract_all_xs(dataset)

    T = 1
    for i in range(T):
        print("ROUND:{}".format(i+1))
        setup_seed(seed)
        model = Network(view, dims, args.feature_dim, class_num, device)
        print(model)
        model = model.to(device)
        state = model.state_dict()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        best_acc, best_nmi, best_pur = 0, 0, 0
        data = dataset[0]


        for epoch in range(args.pre_epochs):
            pretrain(epoch)
        acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=True)
        save_center(model, device, dataset, view, data_size, class_num)


        for epoch in range(args.kl_epochs):
            KL_train(epoch)
            acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)
            if acc > best_acc:
                best_acc, best_nmi, best_pur = acc, nmi, pur
                state = model.state_dict()
                torch.save(state, './models/' + args.dataset + '.pth')

        acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=True)

        # The final result
        accs.append(best_acc)
        nmis.append(best_nmi)
        purs.append(best_pur)
        print('The best clustering performace: ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(best_acc, best_nmi, best_pur))

