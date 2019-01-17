import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, BiGraphConv
from utils import euclidean_dist, normalize

# baseline Bipartite Iterpretable GCN
class BiInterGCN(nn.Module):

    def __init__(self, ts_out, motif_in, motif_out, nclass, dropout, use_att=True):
        super(BiInterGCN, self).__init__()

        ts_hid, motif_hid = 600, 600
        ts_hid2, motif_hid2 = 400, 400

        self.nclass = nclass
        self.gc1 = BiGraphConv(ts_hid, motif_in, motif_hid)
        self.gc2 = BiGraphConv(ts_hid2, motif_hid, motif_hid2)
        self.gc3 = BiGraphConv(ts_out, motif_hid2, motif_out)

        self.ts_bn = nn.BatchNorm1d(ts_hid)
        self.ts_bn2 = nn.BatchNorm1d(ts_hid2)

        self.motif_bn = nn.BatchNorm1d(motif_hid)
        self.motif_bn2 = nn.BatchNorm1d(motif_hid2)

        # self.ac1 = AdjCompute(nfeat)
        # self.ac2 = AdjCompute(nhid)
        # #self.adj_W1 = nn.Linear(self.cmnt_length * 4, 1, bias=True)
        #
        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, 32)
        self.dropout = dropout

        # Representation mapping function
        fc_layers = [200, 100, 100]
        self.linear_1 = nn.Linear(ts_out, fc_layers[0])
        self.linear_2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.linear_3 = nn.Linear(fc_layers[1], fc_layers[2])
        self.bn_1 = nn.BatchNorm1d(fc_layers[0])
        self.bn_2 = nn.BatchNorm1d(fc_layers[1])

        # Attention
        self.use_att = use_att
        if self.use_att:
            D = 128
            self.attention = nn.Sequential(
                nn.Linear(fc_layers[-1], D),
                nn.Tanh(),
                nn.Linear(D, 1)
            )
            # self.att_w = nn.Linear(D, 1)
            # self.att_v = nn.Linear(D, fc_layers[-1])


    def forward(self, input):

        # adj is N * M adjacency matrix, where N is time series size and M is motif size
        adj, motif, labels, idx_train = input

        # x is N * D, where D is the feature dimension
        ts, motif = self.gc1(motif, adj)
        ts, motif = self.ts_bn(ts), self.motif_bn(motif)
        ts, motif = F.leaky_relu(ts), F.leaky_relu(motif)

        ts, motif = self.gc2(motif, adj)
        ts, motif = self.ts_bn2(ts), self.motif_bn2(motif)
        ts, motif = F.leaky_relu(ts), F.leaky_relu(motif)

        ts, motif = self.gc3(motif, adj)

        # linear
        ts = self.linear_1(ts)
        ts = self.bn_1(ts)
        ts = F.leaky_relu(ts)

        ts = self.linear_2(ts)
        ts = self.bn_2(ts)
        ts = F.leaky_relu(ts)

        ts = self.linear_3(ts)

        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze(1) == i).nonzero().squeeze(1)
            if self.use_att:
                A = self.attention(ts[idx_train][idx])  # N_k * 1
                A = torch.transpose(A, 1, 0)  # 1 * N_k
                A = F.softmax(A, dim=1)  # softmax over N_k
                class_repr = torch.mm(A, ts[idx_train][idx]) # 1 * L
                class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
            else:  # if do not use attention, simply use the mean of training samples with the same labels.
                class_repr = ts[idx_train][idx].mean(0)  # L * 1
            proto_list.append(class_repr.view(1, -1))
        x_proto = torch.cat(proto_list, dim=0)
        #print(x_proto)
        dists = euclidean_dist(ts, x_proto)
        #log_dists = F.log_softmax(-dists * 1e7, dim=1)
        return -dists


# baseline FGCN
class InterGCN(nn.Module):

    def __init__(self, ts_out, motif_in, motif_out, nclass, dropout, use_att=True):
        super(InterGCN, self).__init__()

        ts_hid, motif_hid = 600, 600
        ts_hid2, motif_hid2 = 400, 400

        self.nclass = nclass
        self.gc1 = GraphConvolution(motif_in, motif_hid)
        self.gc2 = GraphConvolution(motif_hid, motif_hid2)
        self.gc2 = GraphConvolution(motif_hid2, motif_out)

        self.motif_bn = nn.BatchNorm1d(motif_hid)
        self.motif_bn2 = nn.BatchNorm1d(motif_hid2)

        self.dropout = dropout

        # Representation mapping function
        fc_layers = [200, 100, 100]
        self.linear_1 = nn.Linear(ts_out, fc_layers[0])
        self.linear_2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.linear_3 = nn.Linear(fc_layers[1], fc_layers[2])
        self.bn_1 = nn.BatchNorm1d(fc_layers[0])
        self.bn_2 = nn.BatchNorm1d(fc_layers[1])

        # Attention
        self.use_att = use_att
        if self.use_att:
            D = 128
            self.attention = nn.Sequential(
                nn.Linear(fc_layers[-1], D),
                nn.Tanh(),
                nn.Linear(D, 1)
            )
            # self.att_w = nn.Linear(D, 1)
            # self.att_v = nn.Linear(D, fc_layers[-1])


    def forward(self, input):

        # adj is N * M adjacency matrix, where N is time series size and M is motif size
        adj, motif, labels, idx_train = input

        # x is N * D, where D is the feature dimension
        ts, motif = self.gc1(motif, adj)
        ts, motif = self.ts_bn(ts), self.motif_bn(motif)
        ts, motif = F.leaky_relu(ts), F.leaky_relu(motif)

        ts, motif = self.gc2(motif, adj)
        ts, motif = self.ts_bn2(ts), self.motif_bn2(motif)
        ts, motif = F.leaky_relu(ts), F.leaky_relu(motif)

        ts, motif = self.gc3(motif, adj)

        # linear
        ts = self.linear_1(ts)
        ts = self.bn_1(ts)
        ts = F.leaky_relu(ts)

        ts = self.linear_2(ts)
        ts = self.bn_2(ts)
        ts = F.leaky_relu(ts)

        ts = self.linear_3(ts)

        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze(1) == i).nonzero().squeeze(1)
            if self.use_att:
                A = self.attention(ts[idx_train][idx])  # N_k * 1
                A = torch.transpose(A, 1, 0)  # 1 * N_k
                A = F.softmax(A, dim=1)  # softmax over N_k
                class_repr = torch.mm(A, ts[idx_train][idx]) # 1 * L
                class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
            else:  # if do not use attention, simply use the mean of training samples with the same labels.
                class_repr = ts[idx_train][idx].mean(0)  # L * 1
            proto_list.append(class_repr.view(1, -1))
        x_proto = torch.cat(proto_list, dim=0)
        #print(x_proto)
        dists = euclidean_dist(ts, x_proto)
        #log_dists = F.log_softmax(-dists * 1e7, dim=1)
        return -dists