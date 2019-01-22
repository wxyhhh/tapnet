import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, BiGraphConv
from utils import euclidean_dist, normalize


class TapNet(nn.Module):

    def __init__(self, nfeat, nclass, dropout, layers, use_att=True, use_ss=False, use_metric=False):
        super(TapNet, self).__init__()

        self.nclass = nclass
        self.dropout = dropout
        self.use_metric = use_metric

        # Representation mapping function
        layers = [nfeat] + layers
        self.mapping = nn.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add_module("fc_" + str(i), nn.Linear(layers[i], layers[i + 1]))
            self.mapping.add_module("bn_" + str(i), nn.BatchNorm1d(layers[i + 1]))
            self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.mapping.add_module("fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add_module("bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1]))

        # Attention
        self.use_att = use_att
        if self.use_att:
            self.att_models = nn.ModuleList()
            for _ in range(nclass):
                D = 128
                att_model = nn.Sequential(
                    nn.Linear(layers[-1], D),
                    nn.Tanh(),
                    nn.Linear(D, 1)
                )
                self.att_models.append(att_model)

        # if self.use_att:
        #     D = 128
        #     self.attention = nn.Sequential(
        #         nn.Linear(fc_layers[-1], D),
        #         nn.Tanh(),
        #         nn.Linear(D, 1)
        #     )

        self.use_ss = use_ss  # whether to use semi-supervised mode
        if self.use_ss:
            self.semi_att = nn.Sequential(
                nn.Linear(layers[-1], D),
                nn.Tanh(),
                nn.Linear(D, self.nclass)
            )

    def forward(self, input):
        x, labels, idx_train, idx_val, idx_test = input  # x is N * L, where L is the time-series feature dimension

        # linear mapping to low-dimensional space
        x = self.mapping(x)

        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze(1) == i).nonzero().squeeze(1)
            if self.use_att:
                #A = self.attention(x[idx_train][idx])  # N_k * 1
                A = self.att_models[i](x[idx_train][idx])  # N_k * 1
                A = torch.transpose(A, 1, 0)  # 1 * N_k
                A = F.softmax(A, dim=1)  # softmax over N_k
                #print(A)
                class_repr = torch.mm(A, x[idx_train][idx]) # 1 * L
                class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
            else:  # if do not use attention, simply use the mean of training samples with the same labels.
                class_repr = x[idx_train][idx].mean(0)  # L * 1
            proto_list.append(class_repr.view(1, -1))
        x_proto = torch.cat(proto_list, dim=0)
        #print(x_proto)
        #dists = euclidean_dist(x, x_proto)
        #log_dists = F.log_softmax(-dists * 1e7, dim=1)

        # prototype distance
        proto_dists = euclidean_dist(x_proto, x_proto)
        num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
        proto_dist = torch.sum(proto_dists) / num_proto_pairs

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
        return -dists, proto_dist


