# -*- coding: utf-8 -*-
# @Time    : 2025/4/17 13:19
# @Author  : Ginger
# @FileName: utils.py.py
# @Software: PyCharm
import torch
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def target_distribution_multi_view(qs):
    ps = []
    for q in qs:
        weight = q**2 / q.sum(0)
        p = (weight.t() / weight.sum(1)).t()
        ps.append(p)
    return ps  # 返回 list，每个元素是一个视图的目标分布 [N, K]


def extract_all_xs(dataset):
    num_views = len(dataset[0][0])  # dataset[0][0] 是一个 list，比如 [x1, x2]
    view_arrays = [[] for _ in range(num_views)]

    for i in range(len(dataset)):
        xs, _, _ = dataset[i]  # xs 是 [tensor(x1), tensor(x2), ...]
        for v in range(num_views):
            view_arrays[v].append(xs[v].numpy())  # 先转为 ndarray

    # 把每个 view 的 list 拼成 ndarray: [num_samples, dim]
    view_arrays = [np.stack(view_data, axis=0) for view_data in view_arrays]

    return view_arrays

# 手动存储预训练的聚类中心
def save_center(model, device, dataset, view, data_size, class_num):
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    loader = DataLoader(
            dataset,
            batch_size=data_size,
            shuffle=False,
        )
    model.eval()

    for step, (xs, y, _) in enumerate(loader):
        # for v in range(view):
        #     xs[v] = xs[v].to(device)
        if not isinstance(xs, list):  # 如果是单视图数据
            xs = [xs]  # 转换为单元素列表
        xs = [x.to(device) for x in xs]
        with torch.no_grad():
            xrs, zs, qs = model(xs)
        for v in range(view):
            zv = zs[v].detach()
            kmeans.fit_predict(zv.cpu().detach().data.numpy())
            model.clusteringLayers[v].centroids.data = torch.tensor(kmeans.cluster_centers_).to(device)



