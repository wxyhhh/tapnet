import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, BiGraphConv
from utils import euclidean_dist, normalize


# baseline ProtoGCN
class ProtoGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, use_att=True, use_ss=False):
        super(ProtoGCN, self).__init__()

        self.nclass = nclass
        # self.ac1 = AdjCompute(nfeat)
        # self.ac2 = AdjCompute(nhid)
        # #self.adj_W1 = nn.Linear(self.cmnt_length * 4, 1, bias=True)
        #
        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, 32)
        self.dropout = dropout

        # Representation mapping function
        fc_layers = [2000, 1000, 300]
        self.linear_1 = nn.Linear(nfeat, fc_layers[0])
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

        self.use_ss = use_ss  # whether to use semi-supervised mode
        if self.use_ss:
            self.semi_att = nn.Sequential(
                nn.Linear(fc_layers[-1], D),
                nn.Tanh(),
                nn.Linear(D, self.nclass)
            )

    def forward(self, input):
        x, labels, idx_train, idx_val, idx_test = input  # x is N * L, where L is the time-series feature dimension

        # GCN based representation function
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
            if self.use_att:
                A = self.attention(x[idx_train][idx])  # N_k * 1
                A = torch.transpose(A, 1, 0)  # 1 * N_k
                A = F.softmax(A, dim=1)  # softmax over N_k
                class_repr = torch.mm(A, x[idx_train][idx]) # 1 * L
                class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
            else:  # if do not use attention, simply use the mean of training samples with the same labels.
                class_repr = x[idx_train][idx].mean(0)  # L * 1
            proto_list.append(class_repr.view(1, -1))
        x_proto = torch.cat(proto_list, dim=0)
        #print(x_proto)
        #dists = euclidean_dist(x, x_proto)
        #log_dists = F.log_softmax(-dists * 1e7, dim=1)

        if self.use_ss:
            semi_A = self.semi_att(x[idx_test])  # N_test * c
            semi_A = torch.transpose(semi_A, 1, 0)  # 1 * N_k
            semi_A = F.softmax(semi_A, dim=1)  # softmax over N_k
            x_proto_test = torch.mm(semi_A, x[idx_test])  # c * L
            x_proto = (x_proto + x_proto_test) / 2

            # solution 2
            # row_sum = 1 / torch.sum(-dists[idx_test,], dim=1)
            # row_sum = row_sum.unsqueeze(1).repeat(1, 2)
            # prob = torch.mul(-dists[idx_test,], row_sum)
            # x_proto_test = torch.transpose(torch.mm(torch.transpose(x[idx_test,], 0, 1), prob), 0, 1)

        dists = euclidean_dist(x, x_proto)
        return -dists



# Semi-supervised Prototypical Model
# class SemiProtoGCN(nn.Module):
#
#     def __init__(self, nfeat, nhid, nclass, dropout, use_att=True, use_ss=True):
#         super(SemiProtoGCN, self).__init__()
#
#         self.nclass = nclass
#         # self.ac1 = AdjCompute(nfeat)
#         # self.ac2 = AdjCompute(nhid)
#         # #self.adj_W1 = nn.Linear(self.cmnt_length * 4, 1, bias=True)
#         #
#         # self.gc1 = GraphConvolution(nfeat, nhid)
#         # self.gc2 = GraphConvolution(nhid, 32)
#         self.dropout = dropout
#
#         # Representation mapping function
#         fc_layers = [2000, 1000, 300]
#         self.linear_1 = nn.Linear(nfeat, fc_layers[0])
#         self.linear_2 = nn.Linear(fc_layers[0], fc_layers[1])
#         self.linear_3 = nn.Linear(fc_layers[1], fc_layers[2])
#         self.bn_1 = nn.BatchNorm1d(fc_layers[0])
#         self.bn_2 = nn.BatchNorm1d(fc_layers[1])
#
#         # Attention
#         self.use_att = use_att
#         if self.use_att:
#             D = 128
#             self.attention = nn.Sequential(
#                 nn.Linear(fc_layers[-1], D),
#                 nn.Tanh(),
#                 nn.Linear(D, 1)
#             )
#
#         self.semi_att = nn.Sequential(
#             nn.Linear(fc_layers[-1], D),
#             nn.Tanh(),
#             nn.Linear(D, self.nclass)
#         )
#
#     def forward(self, input):
#         ts, labels, idx_train, idx_val, idx_test = input  # x is N * L, where L is the time-series feature dimension
#
#         # GCN based representation function
#         # adj = self.ac1(x)  # N * N
#         # x = F.relu(self.gc1(x, adj))  # N * hidden_feat
#         # #x = F.dropout(x, self.dropout, training=self.training)  # N * hidden_feat
#         #
#         # adj = self.ac2(x)  # N * N
#         # x = self.gc2(x, adj)
#
#         # linear
#         ts = self.linear_1(ts)
#         ts = self.bn_1(ts)
#         ts = F.leaky_relu(ts)
#
#         ts = self.linear_2(ts)
#         ts = self.bn_2(ts)
#         ts = F.leaky_relu(ts)
#
#         ts = self.linear_3(ts)
#
#         # generate the class protocal with dimension C * D (nclass * dim)
#         proto_list = []
#         for i in range(self.nclass):
#             idx = (labels[idx_train].squeeze(1) == i).nonzero().squeeze(1)
#             if self.use_att:
#                 A = self.attention(ts[idx_train][idx])  # N_k * 1
#                 A = torch.transpose(A, 1, 0)  # 1 * N_k
#                 A = F.softmax(A, dim=1)  # softmax over N_k
#                 class_repr = torch.mm(A, ts[idx_train][idx]) # 1 * L
#                 class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
#             else:  # if do not use attention, simply use the mean of training samples with the same labels.
#                 class_repr = ts[idx_train][idx].mean(0)  # L * 1
#             proto_list.append(class_repr.view(1, -1))
#         x_proto = torch.cat(proto_list, dim=0)
#         #print(x_proto)
#         dists = euclidean_dist(ts, x_proto)
#
#         if self.use_ss:
#             semi_A = self.semi_att(ts[idx_test])  # N_test * c
#             semi_A = torch.transpose(semi_A, 1, 0)  # 1 * N_k
#             semi_A = F.softmax(semi_A, dim=1)  # softmax over N_k
#             x_proto_test = torch.mm(semi_A, ts[idx_test])  # c * L
#         else:
#             # semi-supervised settings
#             # 1. normalize the distance
#             #prob = F.softmax(-dists[idx_test, ])
#             row_sum = 1 / torch.sum(-dists[idx_test,], dim=1)
#             row_sum = row_sum.unsqueeze(1).repeat(1, 2)
#             prob = torch.mul(-dists[idx_test,], row_sum)
#             x_proto_test = torch.transpose(torch.mm(torch.transpose(ts[idx_test,], 0, 1), prob), 0, 1)
#
#
#         x_proto = (x_proto + x_proto_test) / 2
#         dists = euclidean_dist(ts, x_proto)
#         return -dists


