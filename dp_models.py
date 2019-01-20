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



# BiGCN Model for Semi-supervised time series classification
class BiGCN(nn.Module):

    def __init__(self, ts_out, motif_in, motif_out, nclass, dropout):
        super(BiGCN, self).__init__()
        ts_hid, motif_hid = 400, 400
        ts_hid2, motif_hid2 = 100, 100

        self.nclass = nclass

        # self.gc1 = BiGraphConv(ts_hid, motif_in, motif_hid)
        # self.gc2 = BiGraphConv(ts_out, motif_hid, motif_out)

        self.gc1 = BiGraphConv(ts_hid, motif_in, motif_hid)
        self.gc2 = BiGraphConv(ts_hid2, motif_hid, motif_hid2)
        self.gc3 = BiGraphConv(ts_out, motif_hid2, motif_out)

        self.ts_bn = nn.BatchNorm1d(ts_hid)
        self.ts_bn2 = nn.BatchNorm1d(ts_hid2)

        self.motif_bn = nn.BatchNorm1d(motif_hid)
        self.motif_bn2 = nn.BatchNorm1d(motif_hid2)

        self.dropout = dropout

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


        # ts, motif = self.gc2(ts, self.adj.transpose(0, 1))


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

        #return ts


# Motif based GCN Model for Semi-supervised time series classification
class MotifGCN(nn.Module):

    def __init__(self, ts_out, motif_in, motif_out, dropout, nclass, motif_gcn=False):
        super(MotifGCN, self).__init__()
        motif_hid = 300

        self.motif_gcn = motif_gcn
        self.nclass = nclass

        self.ac1 = AdjCompute(motif_in)
        self.ac2 = AdjCompute(motif_hid)
        # self.adj_W1 = nn.Linear(self.cmnt_length * 4, 1, bias=True)

        self.gc1 = GraphConvolution(motif_in, motif_hid)
        self.gc2 = GraphConvolution(motif_hid, motif_out)


        #  =============================================
        fc_layers = [2000, 1000, 300]
        self.linear_1 = nn.Linear(motif_in, fc_layers[0])
        self.linear_2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.linear_3 = nn.Linear(fc_layers[1], fc_layers[2])
        self.bn_1 = nn.BatchNorm1d(fc_layers[0])
        self.bn_2 = nn.BatchNorm1d(fc_layers[1])

        ts_layers = [300, 200, 100]
        self.linear_ts1 = nn.Linear(fc_layers[2], ts_layers[0])
        self.linear_ts2 = nn.Linear(ts_layers[0], ts_layers[1])
        self.linear_ts3 = nn.Linear(ts_layers[1], ts_layers[2])
        self.bn_ts1 = nn.BatchNorm1d(ts_layers[0])
        self.bn_ts2 = nn.BatchNorm1d(ts_layers[1])

        # self.ts_bn = nn.BatchNorm1d(ts_hid)
        # self.ts_bn2 = nn.BatchNorm1d(ts_hid2)
        #
        # self.motif_bn = nn.BatchNorm1d(motif_hid)
        # self.motif_bn2 = nn.BatchNorm1d(motif_hid2)

        self.dropout = dropout

    def forward(self, input):
        adj, motif, labels, idx_train = input

        if self.motif_gcn:
            # generate motif embedding by GCN
            adj = self.ac1(motif)  # N * N
            motif = F.relu(self.gc1(motif, adj))  # N * hidden_feat
            # x = F.dropout(x, self.dropout, training=self.training)  # N * hidden_feat

            adj = self.ac2(motif)  # N * N
            motif = self.gc2(motif, adj)
        else:

            # linear
            motif = self.linear_1(motif)
            motif = self.bn_1(motif)
            motif = F.leaky_relu(motif)

            motif = self.linear_2(motif)
            motif = self.bn_2(motif)
            motif = F.leaky_relu(motif)

            motif = self.linear_3(motif)

        ts = torch.mm(adj, motif)


        #
        # ts = self.linear_ts1(ts)
        # ts = self.bn_ts1(ts)
        # ts = F.leaky_relu(ts)
        #
        # ts = self.linear_ts2(ts)
        # ts = self.bn_ts2(ts)
        # ts = F.leaky_relu(ts)
        #
        # ts = self.linear_ts3(ts)

        # generate ts embedding by attention model
        #ts = Att(motif)

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


# baseline ProtoGCNBatch
class ProtoGCNBatch(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, use_att=True):
        super(ProtoGCNBatch, self).__init__()

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


    def forward(self, ts, labels, ts_proto=None):
        # x is N * L, where L is the time-series feature dimension

        # GCN based representation function
        # adj = self.ac1(x)  # N * N
        # x = F.relu(self.gc1(x, adj))  # N * hidden_feat
        # #x = F.dropout(x, self.dropout, training=self.training)  # N * hidden_feat
        #
        # adj = self.ac2(x)  # N * N
        # x = self.gc2(x, adj)

        # linear
        ts = self.linear_1(ts)
        ts = self.bn_1(ts)
        ts = F.leaky_relu(ts)

        ts = self.linear_2(ts)
        ts = self.bn_2(ts)
        ts = F.leaky_relu(ts)

        ts = self.linear_3(ts)

        # generate the class protocal with dimension C * D (nclass * dim)
        if ts_proto is None:
            proto_list = []
            for i in range(self.nclass):
                idx = (labels == i).nonzero().squeeze(1)
                if self.use_att:
                    A = self.attention(ts[idx])  # N_k * 1
                    A = torch.transpose(A, 1, 0)  # 1 * N_k
                    A = F.softmax(A, dim=1)  # softmax over N_k
                    class_repr = torch.mm(A, ts[idx]) # 1 * L
                    class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
                else:  # if do not use attention, simply use the mean of training samples with the same labels.
                    class_repr = ts[idx].mean(0)  # L * 1
                proto_list.append(class_repr.view(1, -1))
            ts_proto = torch.cat(proto_list, dim=0)
        dists = euclidean_dist(ts, ts_proto)
        return ts_proto, -dists

# Time Series Prediction Network
class TPNet(nn.Module):

    def __init__(self, channel, ts_length, nclass, dropout, use_att=True):
        super(TPNet, self).__init__()

        self.channel = channel
        self.ts_length = ts_length
        self.nclass = nclass
        # self.ac1 = AdjCompute(nfeat)
        # self.ac2 = AdjCompute(nhid)
        # #self.adj_W1 = nn.Linear(self.cmnt_length * 4, 1, bias=True)
        #
        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, 32)
        self.dropout = dropout

        # LSTM
        self.lstm_dim = 100
        self.lstm = nn.LSTM(self.ts_length, self.lstm_dim)
        # convolutional layer
        # features for each hidden layers
        out_channels = [512, 512, 256]
        kernels = [8, 5, 3]
        poolings = [2, 2, 2]
        self.conv_1 = nn.Conv1d(self.channel, out_channels[0], kernel_size=kernels[0], stride=1)
        #self.maxpool_1 = nn.MaxPool1d(poolings[0])
        self.bn_1 = nn.BatchNorm1d(out_channels[0])
        self.conv_2 = nn.Conv1d(out_channels[0], out_channels[1], kernel_size=kernels[1], stride=1)
        #self.maxpool_2 = nn.MaxPool1d(poolings[1])
        self.bn_2 = nn.BatchNorm1d(out_channels[1])
        self.conv_3 = nn.Conv1d(out_channels[1], out_channels[2], kernel_size=kernels[2], stride=1)
        #self.maxpool_3 = nn.MaxPool1d(poolings[2])
        self.bn_3 = nn.BatchNorm1d(out_channels[2])
        # self.conv_4 = nn.Conv1d(out_channels[2], out_channels[3], kernel_size=kernels[3], stride=1)
        # self.maxpool_4 = nn.MaxPool1d(poolings[3])
        # self.bn_4 = nn.BatchNorm1d(out_channels[3])



        # self.conv = nn.Sequential(
        #     nn.Conv1d(self.channel, out_channels[0], kernel_size=8, stride=2),
        #     nn.BatchNorm1d(out_channels[0]),
        #     nn.ReLU(),
        #     # nn.Conv1d(nhids[0], nhids[1], kernel_size=5, stride=2),
        #     # nn.BatchNorm1d(nhids[1]),
        #     # nn.ReLU(),
        #     # nn.Conv1d(nhids[1], nhids[2], kernel_size=3, stride=2),
        #     # nn.BatchNorm1d(nhids[1]),
        #     # nn.ReLU()
        # )

        self.linear_init_feature = out_channels[2] * (self.ts_length - sum(kernels) + len(kernels)) + self.lstm_dim * self.channel
        #self.linear_init_feature = out_channels[2] * (5)
        # Representation mapping function
        fc_layers = [2000, 1000, 300]
        self.linear_mapping = nn.Sequential(
            nn.Linear(self.linear_init_feature, fc_layers[0]),
            nn.BatchNorm1d(fc_layers[0]),
            nn.ReLU(),
            nn.Linear(fc_layers[0], fc_layers[1]),
            nn.BatchNorm1d(fc_layers[1]),
            nn.ReLU(),
            nn.Linear(fc_layers[1], fc_layers[2]),
        )

        # Representation mapping function
        # fc_layers = [2000, 1000, 300]
        # self.linear_1 = nn.Linear(nhids[-1], fc_layers[0])
        # self.linear_2 = nn.Linear(fc_layers[0], fc_layers[1])
        # self.linear_3 = nn.Linear(fc_layers[1], fc_layers[2])
        # self.bn_1 = nn.BatchNorm1d(fc_layers[0])
        # self.bn_2 = nn.BatchNorm1d(fc_layers[1])

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
        ts, labels, idx_train = input  # x is N * L, where L is the time-series feature dimension

        # N: sample number
        # C: channel (number of variates in time series)
        # L: length of time series
        N = ts.size(0)

        # LSTM
        ts_lstm = self.lstm(ts)[0]
        ts_lstm = ts_lstm.view(N, -1)

        # Covolutional Network
        # input ts: # N * C * L
        ts = self.conv_1(ts)  # N * C * L
        ts = self.bn_1(ts)
        #ts = self.maxpool_1(ts)
        ts = F.leaky_relu(ts)

        ts = self.conv_2(ts)
        ts = self.bn_2(ts)
        #ts = self.maxpool_2(ts)
        ts = F.leaky_relu(ts)

        ts = self.conv_3(ts)
        ts = self.bn_3(ts)
        #ts = self.maxpool_3(ts)
        ts = F.leaky_relu(ts)

        # ts = self.conv_4(ts)
        # ts = self.bn_4(ts)
        # #ts = self.maxpool_4(ts)
        # ts = F.leaky_relu(ts)
        #print("ts dimension", ts.size())
        ts = ts.view(N, -1)

        ts = torch.cat([ts, ts_lstm], dim=1)

        # linear
        ts = self.linear_mapping(ts)

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