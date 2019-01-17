import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import torch
import pandas as pd


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def loadsparse(filename):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    a = sp.csr_matrix(a)
    return a

def loaddata(filename):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    return a


def load_raw_ts(path, dataset, tensor_format=True):
    path = path + dataset + "/"
    x_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'y_test.npy')

    ts = np.concatenate((x_train, x_test), axis=0)
    ts = np.transpose(ts, axes=(0, 2, 1))
    labels = np.concatenate((y_train, y_test), axis=0)

    nclass = np.amax(labels) + 1

    # total data size: 934
    train_size = y_train.shape[0]
    # train_size = 10
    total_size = labels.shape[0]
    idx_train = range(train_size)
    idx_val = range(train_size, total_size)
    idx_test = range(train_size, total_size)

    if tensor_format:
        # features = torch.FloatTensor(np.array(features))
        ts = torch.FloatTensor(np.array(ts))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return ts, labels, idx_train, idx_val, idx_test, nclass


def load_bigraph_adj(filename, shape):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    row = a[:, 0] - 1
    col = a[:, 1] - 1
    value = a[:, 2]
    # value = [1] * a.shape[0]
    adj = sp.csr_matrix((value, (row, col)), shape=shape).toarray()
    return adj


# load the data generated from MUSE
def load_bigraph(path="./data/", dataset="ECG", tensor_format=True):

    path = path + dataset + "/"
    file_header = dataset.lower() + "_"

    # train_features = loadsparse(path + file_header + "train.csv")
    # test_features = loadsparse(path + file_header + "test.csv")
    # features = sp.vstack([train_features, test_features])
    # features = normalize(features)

    # load motif embedding
    motif_features = loadsparse(path + file_header + "motif_embedding.csv")


    # load labels
    train_labels = loaddata(path + file_header + "train_label.csv")
    test_labels = loaddata(path + file_header + "test_label.csv")
    labels = np.concatenate((train_labels, test_labels), axis=0)

    nclass = np.amax(labels) + 1

    # load the bipartite graph
    bigraph_adj = load_bigraph_adj(path + file_header + "mapping_table.csv",
                                   shape=(labels.shape[0], motif_features.shape[0]))

    # total data size: 934
    train_size = train_labels.shape[0]
    #train_size = 10
    total_size = labels.shape[0]
    idx_train = range(train_size)
    idx_val = range(train_size, total_size)
    idx_test = range(train_size, total_size)

    if tensor_format:
        # features = torch.FloatTensor(np.array(features))
        bigraph_adj = torch.FloatTensor(np.array(bigraph_adj))
        motif_features = torch.FloatTensor(np.array(motif_features.todense()))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return bigraph_adj, motif_features, labels, idx_train, idx_val, idx_test, nclass


# load the data generated from MUSE
def load_muse_data(path="./data/", dataset="ECG", tensor_format=True):

    path = path + dataset + "/"
    file_header = dataset.lower() + "_"

    train_features = loadsparse(path + file_header + "train.csv")
    test_features = loadsparse(path + file_header + "test.csv")
    features = sp.vstack([train_features, test_features])
    features = normalize(features)

    train_labels = loaddata(path + file_header + "train_label.csv")
    test_labels = loaddata(path + file_header + "test_label.csv")
    labels = np.concatenate((train_labels, test_labels), axis=0)

    nclass = np.amax(labels) + 1



    # total data size: 934
    train_size = train_features.shape[0]
    #train_size = 10
    total_size = features.shape[0]
    idx_train = range(train_size)
    idx_val = range(train_size, total_size)
    idx_test = range(train_size, total_size)

    if tensor_format:
        features = torch.FloatTensor(np.array(features))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return features, labels, idx_train, idx_val, idx_test, nclass


def normalize(mx):
    """Row-normalize sparse matrix"""
    # rowsum = np.array(mx.sum(1))
    # r_inv = np.power(rowsum, -1).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    # mx = r_mat_inv.dot(mx)

    row_sums = mx.sum(axis=1)
    mx = mx / row_sums

    return mx


def accuracy(output, labels):

    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

    return accuracy_score


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
