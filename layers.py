import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class BiGraphConv(Module):
    """
    Bipartite GCN layer with two different node types
    """

    def __init__(self, a_features, b_features, bias=True):
        super(BiGraphConv, self).__init__()
        self.a_features = a_features
        self.b_features = b_features

        self.a_weight = Parameter(torch.FloatTensor(b_features, a_features))
        self.b_weight = Parameter(torch.FloatTensor(a_features, b_features))
        if bias:
            self.a_bias = Parameter(torch.FloatTensor(a_features))
            self.b_bias = Parameter(torch.FloatTensor(b_features))
        else:
            self.register_parameter('a_bias', None)
            self.register_parameter('b_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        a_stdv = 1. / math.sqrt(self.a_weight.size(1))
        self.a_weight.data.uniform_(-a_stdv, a_stdv)
        if self.a_bias is not None:
            self.a_bias.data.uniform_(-a_stdv, a_stdv)

        b_stdv = 1. / math.sqrt(self.b_weight.size(1))
        self.b_weight.data.uniform_(-b_stdv, b_stdv)
        if self.b_bias is not None:
            self.b_bias.data.uniform_(-b_stdv, b_stdv)

    def forward(self, b_input, adj):
        a_support = torch.mm(b_input, self.a_weight)
        a_output = torch.spmm(adj, a_support)
        if self.a_bias is not None:
            return a_output + self.a_bias

        b_support = torch.mm(a_output, self.b_weight)
        b_output = torch.spmm(adj.transpose(), b_support)
        if self.b_bias is not None:
            return b_output + self.b_bias

        return a_output, b_output
