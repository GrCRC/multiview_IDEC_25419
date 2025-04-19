import torch.nn as nn
from torch.nn.functional import normalize
import torch
from torch.nn.parameter import Parameter

class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, n_z):
        super(ClusteringLayer, self).__init__()
        self.centroids = Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)       #表示聚类中心的参数矩阵，大小为 (n_clusters, n_z)，可以进行梯度更新

    def forward(self, x):
        q = 1.0 / (1 + torch.sum(torch.pow(x.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        return q

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num, device):
        super(Network, self).__init__()
        self.view = view
        self.encoders = []
        self.decoders = []
        self.clusteringLayers = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
            self.clusteringLayers.append(ClusteringLayer(class_num, feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.clusteringLayers = nn.ModuleList(self.clusteringLayers)


    def forward(self, xs, zs_gradient=True):

        xrs = []
        zs = []
        qs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.clusteringLayers[v](z)
            xr = self.decoders[v](z)

            zs.append(z)
            xrs.append(xr)
            qs.append(q)


        return xrs,zs,qs

