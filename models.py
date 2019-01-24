import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, BiGraphConv
from utils import euclidean_dist, normalize, output_conv_size


class TapNet(nn.Module):

    def __init__(self, nfeat, len_ts, nclass, dropout, filters, kernels, layers, use_att=True, use_ss=False, use_metric=False,
                 use_muse=False, use_lstm=False, use_cnn=True):
        super(TapNet, self).__init__()

        self.nclass = nclass
        self.dropout = dropout
        self.use_metric = use_metric
        self.use_muse = use_muse
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        if not self.use_muse:
            # LSTM
            self.channel = nfeat
            self.ts_length = len_ts

            self.lstm_dim = 128
            self.lstm = nn.LSTM(self.ts_length, self.lstm_dim)
            #self.lstm = nn.LSTM(self.channel, self.lstm_dim)
            #self.dropout = nn.Dropout(0.8)

            # convolutional layer
            # features for each hidden layers
            #out_channels = [256, 128, 256]
            #filters = [256, 256, 128]
            poolings = [2, 2, 2]
            paddings = [0, 0, 0]
            self.conv_1 = nn.Conv1d(self.channel, filters[0], kernel_size=kernels[0], stride=1, padding=paddings[0])
            # self.maxpool_1 = nn.MaxPool1d(poolings[0])
            self.conv_bn_1 = nn.BatchNorm1d(filters[0])
            self.conv_2 = nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], stride=1, padding=paddings[1])
            # self.maxpool_2 = nn.MaxPool1d(poolings[1])
            self.conv_bn_2 = nn.BatchNorm1d(filters[1])
            self.conv_3 = nn.Conv1d(filters[1], filters[2], kernel_size=kernels[2], stride=1, padding=paddings[2])
            # self.maxpool_3 = nn.MaxPool1d(poolings[2])
            self.conv_bn_3 = nn.BatchNorm1d(filters[2])
            # self.conv_4 = nn.Conv1d(out_channels[2], out_channels[3], kernel_size=kernels[3], stride=1)
            # self.maxpool_4 = nn.MaxPool1d(poolings[3])
            # self.bn_4 = nn.BatchNorm1d(out_channels[3])

            # compute the size of input for fully connected layers
            fc_input = 0
            if self.use_cnn:
                conv_size = len_ts
                for i in range(len(filters)):
                    conv_size = output_conv_size(conv_size, kernels[i], 1, paddings[i])
                fc_input += conv_size * filters[-1]
            if self.use_lstm:
                fc_input += self.channel * self.lstm_dim

            # fc_input = 0
            # if self.use_cnn:
            #     conv_size = len_ts
            #     for i in range(len(out_channels)):
            #         conv_size = output_conv_size(conv_size, kernels[i], 1, 0)
            #     fc_input += conv_size
            # if self.use_lstm:
            #     fc_input += self.lstm_dim

        # Representation mapping function
        layers = [fc_input] + layers
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

        if not self.use_muse:
            N = x.size(0)

            # LSTM
            if self.use_lstm:
                x_lstm = self.lstm(x)[0]
                # x_lstm = self.dropout(x_lstm)
                #x_lstm = torch.mean(x_lstm, 1)
                x_lstm = x_lstm.view(N, -1)

            if self.use_cnn:
                # Covolutional Network
                # input ts: # N * C * L
                x_conv = x
                x_conv = self.conv_1(x_conv)  # N * C * L
                x_conv = self.conv_bn_1(x_conv)
                # ts = self.maxpool_1(ts)
                x_conv = F.leaky_relu(x_conv)

                x_conv = self.conv_2(x_conv)
                x_conv = self.conv_bn_2(x_conv)
                # ts = self.maxpool_2(ts)
                x_conv = F.leaky_relu(x_conv)

                x_conv = self.conv_3(x_conv)
                x_conv = self.conv_bn_3(x_conv)
                # ts = self.maxpool_3(ts)
                x_conv = F.leaky_relu(x_conv)

                #x_conv = torch.mean(x_conv, 1)
                #aa = F.avg_pool1d(x_conv, kernel_size=3)

                x_conv = x_conv.view(N, -1)

            if self.use_lstm and self.use_cnn:
                x = torch.cat([x_conv, x_lstm], dim=1)
            elif self.use_lstm:
                x = x_lstm
            elif self.use_cnn:
                x = x_conv
            #

        # linear mapping to low-dimensional space
        x = self.mapping(x)

        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
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


