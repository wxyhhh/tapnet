import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, BiGraphConv
from utils import euclidean_dist, normalize, output_conv_size


class TapNet(nn.Module):

    def __init__(self, nfeat, len_ts, nclass, dropout, layers, use_att=True, use_ss=False, use_metric=False, use_raw=True):
        super(TapNet, self).__init__()

        self.nclass = nclass
        self.dropout = dropout
        self.use_metric = use_metric
        self.use_raw = use_raw
        if self.use_raw:
            # LSTM
            self.channel = nfeat
            self.ts_length = len_ts

            self.lstm_dim = 100
            self.lstm = nn.LSTM(self.ts_length, self.lstm_dim)
            # convolutional layer
            # features for each hidden layers
            out_channels = [256, 256, 128]
            kernels = [8, 5, 3]
            poolings = [2, 2, 2]
            self.conv_1 = nn.Conv1d(self.channel, out_channels[0], kernel_size=kernels[0], stride=1)
            # self.maxpool_1 = nn.MaxPool1d(poolings[0])
            self.conv_bn_1 = nn.BatchNorm1d(out_channels[0])
            self.conv_2 = nn.Conv1d(out_channels[0], out_channels[1], kernel_size=kernels[1], stride=1)
            # self.maxpool_2 = nn.MaxPool1d(poolings[1])
            self.conv_bn_2 = nn.BatchNorm1d(out_channels[1])
            self.conv_3 = nn.Conv1d(out_channels[1], out_channels[2], kernel_size=kernels[2], stride=1)
            # self.maxpool_3 = nn.MaxPool1d(poolings[2])
            self.conv_bn_3 = nn.BatchNorm1d(out_channels[2])
            # self.conv_4 = nn.Conv1d(out_channels[2], out_channels[3], kernel_size=kernels[3], stride=1)
            # self.maxpool_4 = nn.MaxPool1d(poolings[3])
            # self.bn_4 = nn.BatchNorm1d(out_channels[3])

            in_size = len_ts
            for i in range(len(out_channels)):
                in_size = output_conv_size(in_size, kernels[i], 1, 0)
            nfeat = in_size * out_channels[-1] + self.channel * self.lstm_dim

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

        if self.use_raw:
            N = x.size(0)

            # LSTM
            x_lstm = self.lstm(x)[0]
            x_lstm = x_lstm.view(N, -1)

            # Covolutional Network
            # input ts: # N * C * L
            x = self.conv_1(x)  # N * C * L
            x = self.conv_bn_1(x)
            # ts = self.maxpool_1(ts)
            x = F.leaky_relu(x)

            x = self.conv_2(x)
            x = self.conv_bn_2(x)
            # ts = self.maxpool_2(ts)
            x = F.leaky_relu(x)

            x = self.conv_3(x)
            x = self.conv_bn_3(x)
            # ts = self.maxpool_3(ts)
            x = F.leaky_relu(x)

            # ts = self.conv_4(ts)
            # ts = self.bn_4(ts)
            # #ts = self.maxpool_4(ts)
            # ts = F.leaky_relu(ts)
            # print("ts dimension", ts.size())
            x = x.view(N, -1)

            x = torch.cat([x, x_lstm], dim=1)

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


