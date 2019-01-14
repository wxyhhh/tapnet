import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from utils import euclidean_dist, normalize

class AdjCompute(nn.Module):
    def __init__(self, num_feat, nf=16, ratio=[1, 1, 0.5, 0.5], use_conv=True):
        super(AdjCompute, self).__init__()

        self.use_conv = use_conv

        # features for each hidden layers
        num_hidden_feats = [int(nf * i) for i in ratio]

        if self.use_conv:
            self.conv2d_1 = nn.Conv2d(num_feat, num_hidden_feats[0], 1, stride=1)
            self.conv_bn_1 = nn.BatchNorm2d(num_hidden_feats[0])

            self.conv2d_2 = nn.Conv2d(num_hidden_feats[0], num_hidden_feats[1], 1, stride=1)
            self.conv_bn_2 = nn.BatchNorm2d(num_hidden_feats[1])

            self.conv2d_3 = nn.Conv2d(num_hidden_feats[1], num_hidden_feats[2], 1, stride=1)
            self.conv_bn_3 = nn.BatchNorm2d(num_hidden_feats[2])

            self.conv2d_4 = nn.Conv2d(num_hidden_feats[2], num_hidden_feats[3], 1, stride=1)
            self.conv_bn_4 = nn.BatchNorm2d(num_hidden_feats[3])

            self.conv2d_last = nn.Conv2d(num_hidden_feats[3], 1, 1, stride=1)
        else:
            self.linear_1 = nn.Linear(num_feat, num_hidden_feats[0])
            self.linear_bn_1 = nn.BatchNorm2d(num_hidden_feats[0])

            self.linear_2 = nn.Linear(num_hidden_feats[0], num_hidden_feats[1])
            self.linear_bn_2 = nn.BatchNorm2d(num_hidden_feats[1])

            self.linear_3 = nn.Linear(num_hidden_feats[1], num_hidden_feats[2])
            self.linear_bn_3 = nn.BatchNorm2d(num_hidden_feats[2])

            self.linear_4 = nn.Linear(num_hidden_feats[2], num_hidden_feats[3])
            self.linear_bn_4 = nn.BatchNorm2d(num_hidden_feats[3])

            self.linear_last = nn.Linear(num_hidden_feats[3], 1)

    def forward(self, x):
        # x is N * D, where D is the feature dimension
        a1 = x.unsqueeze(1)  # N * 1 * D
        a2 = torch.transpose(a1, 0, 1)  # 1 * N * D
        adj = torch.abs(a1 - a2)  # N * N * D

        if self.use_conv:
            adj = torch.transpose(adj, 0, 2) # D * N * N
            adj = adj.unsqueeze(0)  # 1 * D * N * N

            adj = self.conv2d_1(adj)  # 1 * D_1 * N * N
            adj = self.conv_bn_1(adj)
            adj = F.leaky_relu(adj)

            adj = self.conv2d_2(adj)  # 1 * D_2 * N * N
            adj = self.conv_bn_2(adj)
            adj = F.leaky_relu(adj)

            adj = self.conv2d_3(adj)  # 1 * D_3 * N * N
            adj = self.conv_bn_3(adj)
            adj = F.leaky_relu(adj)

            adj = self.conv2d_4(adj)  # 1 * D_4 * N * N
            adj = self.conv_bn_4(adj)
            adj = F.leaky_relu(adj)

            adj = self.conv2d_last(adj)  # 1 * 1 * N * N
            adj = adj.squeeze(0)
            adj = adj.squeeze(0)

        else:

            adj = self.linear_1(adj)  # N * N * D
            #adj = self.linear_bn_1(adj)
            adj = F.leaky_relu(adj)

            adj = self.linear_2(adj)
            #adj = self.linear_bn_2(adj)
            adj = F.leaky_relu(adj)

            adj = self.linear_3(adj)
            #adj = self.linear_bn_3(adj)
            adj = F.leaky_relu(adj)

            adj = self.linear_4(adj)
            #adj = self.linear_bn_4(adj)
            adj = F.leaky_relu(adj)

            adj = self.linear_last(adj)  # N * N * 1

            adj = adj.squeeze(2)

        return adj

# baseline FGCN
class FGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(FGCN, self).__init__()

        self.layer_num = 2
        self.ac1 = AdjCompute(nfeat)
        self.ac2 = AdjCompute(nhid)
        #self.adj_W1 = nn.Linear(self.cmnt_length * 4, 1, bias=True)

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):

        # x is N * D, where D is the feature dimension
        adj = self.ac1(x)  # N * N
        x = F.relu(self.gc1(x, adj))  # N * hidden_feat
        #x = F.dropout(x, self.dropout, training=self.training)  # N * hidden_feat

        adj = self.ac2(x)  # N * N
        x = self.gc2(x, adj)
        return x

# baseline FGCN
class ProtoGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ProtoGCN, self).__init__()

        self.nclass = nclass
        self.layer_num = 2
        self.ac1 = AdjCompute(nfeat)
        self.ac2 = AdjCompute(nhid)
        #self.adj_W1 = nn.Linear(self.cmnt_length * 4, 1, bias=True)

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 32)
        self.dropout = dropout

        self.linear_1 = nn.Linear(nfeat, 100)
        self.linear_2 = nn.Linear(100, 80)
        self.linear_3 = nn.Linear(80, 50)

    def forward(self, x, labels, idx_train):

        # x is N * D, where D is the feature dimension
        adj = self.ac1(x)  # N * N
        x = F.relu(self.gc1(x, adj))  # N * hidden_feat
        #x = F.dropout(x, self.dropout, training=self.training)  # N * hidden_feat

        adj = self.ac2(x)  # N * N
        x = self.gc2(x, adj)

        # x = self.linear_1(x)
        #
        # x = F.relu(x)
        #
        # x = self.linear_2(x)
        # x = F.relu(x)
        #
        # x = self.linear_3(x)
        #x = F.leaky_relu(x)

        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze(1) == i).nonzero().squeeze(1)
            class_repr = x[idx_train][idx].mean(0)
            proto_list.append(class_repr.view(1, -1))
        x_proto = torch.cat(proto_list, dim=0)
        #print(x_proto)
        dists = euclidean_dist(x, x_proto) * 1e7
        #log_dists = F.log_softmax(-dists * 1e7, dim=1)
        return -dists


# Motif based GCN Model for Semi-supervised time series classification
class MotifGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MotifGCN, self).__init__()

        self.motif_layer_num = 2
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout


    def forward(self, x, adj):

        # two layers GCNs for motif
        for _ in range(self.motif_layer_num):
            adj_motif = 1
            embed_motif = 2



        #
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
