import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, BiGraphConv
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

    def forward(self, input):
        x = input[0]
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
        # self.ac1 = AdjCompute(nfeat)
        # self.ac2 = AdjCompute(nhid)
        # #self.adj_W1 = nn.Linear(self.cmnt_length * 4, 1, bias=True)
        #
        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, 32)
        self.dropout = dropout

        fc_layers = [2000, 1000, 300]
        self.linear_1 = nn.Linear(nfeat, fc_layers[0])
        self.linear_2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.linear_3 = nn.Linear(fc_layers[1], fc_layers[2])
        self.bn_1 = nn.BatchNorm1d(fc_layers[0])
        self.bn_2 = nn.BatchNorm1d(fc_layers[1])

    def forward(self, input):
        x, labels, idx_train = input
        # x is N * D, where D is the feature dimension
        # adj = self.ac1(x)  # N * N
        # x = F.relu(self.gc1(x, adj))  # N * hidden_feat
        # #x = F.dropout(x, self.dropout, training=self.training)  # N * hidden_feat
        #
        # adj = self.ac2(x)  # N * N
        # x = self.gc2(x, adj)

        # linear
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = F.leaky_relu(x)

        x = self.linear_2(x)
        x = self.bn_2(x)
        x = F.leaky_relu(x)

        x = self.linear_3(x)

        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze(1) == i).nonzero().squeeze(1)
            class_repr = x[idx_train][idx].mean(0)
            proto_list.append(class_repr.view(1, -1))
        x_proto = torch.cat(proto_list, dim=0)
        #print(x_proto)
        dists = euclidean_dist(x, x_proto)
        #log_dists = F.log_softmax(-dists * 1e7, dim=1)
        return -dists


# BiGCN Model for Semi-supervised time series classification
class BiGCN(nn.Module):

    def __init__(self, adj, ts_out, motif_in, motif_out, nclass, dropout):
        super(BiGCN, self).__init__()
        ts_hid, motif_hid = 400, 400
        ts_hid2, motif_hid2 = 100, 100

        self.nclass = nclass
        self.adj = adj  # N * M, where N is time series size and M is motif size

        # self.gc1 = BiGraphConv(ts_hid, motif_in, motif_hid)
        # self.gc2 = BiGraphConv(ts_out, motif_hid, motif_out)

        self.gc1 = BiGraphConv(ts_hid, motif_in, motif_hid)
        self.gc2 = BiGraphConv(ts_hid2, motif_hid, motif_hid2)
        self.gc3 = BiGraphConv(2, motif_hid2, motif_out)

        self.ts_bn = nn.BatchNorm1d(ts_hid)
        self.ts_bn2 = nn.BatchNorm1d(ts_hid2)

        self.motif_bn = nn.BatchNorm1d(motif_hid)
        self.motif_bn2 = nn.BatchNorm1d(motif_hid2)

        self.dropout = dropout

    def forward(self, input):
        motif, labels, idx_train = input

        # x is N * D, where D is the feature dimension
        ts, motif = self.gc1(motif, self.adj)
        ts, motif = self.ts_bn(ts), self.motif_bn(motif)
        ts, motif = F.leaky_relu(ts), F.leaky_relu(motif)

        ts, motif = self.gc2(motif, self.adj)
        ts, motif = self.ts_bn2(ts), self.motif_bn2(motif)
        ts, motif = F.leaky_relu(ts), F.leaky_relu(motif)

        ts, motif = self.gc3(motif, self.adj)


        # ts, motif = self.gc2(ts, self.adj.transpose(0, 1))


        # generate the class protocal with dimension C * D (nclass * dim)
        # proto_list = []
        # for i in range(self.nclass):
        #     idx = (labels[idx_train].squeeze(1) == i).nonzero().squeeze(1)
        #     class_repr = ts[idx_train][idx].mean(0)
        #     proto_list.append(class_repr.view(1, -1))
        # ts_proto = torch.cat(proto_list, dim=0)
        # # print(x_proto)
        # dists = euclidean_dist(ts, ts_proto)
        # # log_dists = F.log_softmax(-dists * 1e7, dim=1)
        # return -dists

        return ts


# Motif based GCN Model for Semi-supervised time series classification
class MotifGCN(nn.Module):

    def __init__(self, adj, ts_out, motif_in, motif_out, dropout, nclass):
        super(MotifGCN, self).__init__()
        ts_hid, motif_hid = 400, 400
        ts_hid2, motif_hid2 = 100, 100

        self.nclass = nclass
        self.adj = adj

        self.ac1 = AdjCompute(motif_in)
        self.ac2 = AdjCompute(motif_hid)
        # self.adj_W1 = nn.Linear(self.cmnt_length * 4, 1, bias=True)

        self.gc1 = GraphConvolution(motif_in, motif_hid)
        self.gc2 = GraphConvolution(motif_hid, motif_out)

        # self.gc1 = BiGraphConv(ts_hid, motif_in, motif_hid)
        # self.gc2 = BiGraphConv(ts_out, motif_hid, motif_out)

        self.gc1 = BiGraphConv(ts_hid, motif_in, motif_hid)
        self.gc2 = BiGraphConv(ts_hid2, motif_hid, motif_hid2)
        self.gc3 = BiGraphConv(2, motif_hid2, motif_out)

        self.ts_bn = nn.BatchNorm1d(ts_hid)
        self.ts_bn2 = nn.BatchNorm1d(ts_hid2)

        self.motif_bn = nn.BatchNorm1d(motif_hid)
        self.motif_bn2 = nn.BatchNorm1d(motif_hid2)

        self.dropout = dropoutd

    def forward(self, input):
        motif, labels, idx_train = input

        # generate motif embedding by GCN
        adj = self.ac1(motif)  # N * N
        motif = F.relu(self.gc1(motif, adj))  # N * hidden_feat
        # x = F.dropout(x, self.dropout, training=self.training)  # N * hidden_feat

        adj = self.ac2(motif)  # N * N
        motif = self.gc2(motif, adj)

        # generate ts embedding by attention model
        ts = Att(motif)

        # use prototypical network


        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze(1) == i).nonzero().squeeze(1)
            class_repr = ts[idx_train][idx].mean(0)
            proto_list.append(class_repr.view(1, -1))
        ts_proto = torch.cat(proto_list, dim=0)
        # print(x_proto)
        dists = euclidean_dist(ts, ts_proto)
        # log_dists = F.log_softmax(-dists * 1e7, dim=1)
        return -dists