import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import torch
import pandas as pd
import random


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
    val_size = int(train_size * 0.2)
    # idx_train = range(train_size - val_size)
    # idx_val = range(train_size - val_size, train_size)
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


# partition the indices of training, validation and testing data
def partition_data(train_size, val_size, test_size):
    non_test_size = train_size + val_size
    idx_non_test = random.sample(range(non_test_size), non_test_size)
    idx_train = idx_non_test[:train_size]
    idx_val = idx_non_test[train_size: train_size + val_size]
    idx_test = range(non_test_size, non_test_size + test_size)

    return idx_train, idx_val, idx_test


# load the data generated from MUSE
def load_muse_data(data_path="./data/", dataset="ECG", tensor_format=True):

    path = data_path + "muse/" + dataset + "/"
    file_header = dataset.lower() + "_"

    train_features = loadsparse(path + file_header + "train.csv")
    # shuttle train features
    non_test_size = train_features.shape[0]
    idx_non_test = random.sample(range(non_test_size), non_test_size)
    train_features = train_features[idx_non_test, ]

    test_features = loadsparse(path + file_header + "test.csv")
    features = sp.vstack([train_features, test_features])
    features = normalize(features)

    train_labels = loaddata(path + file_header + "train_label.csv")
    train_labels = train_labels[idx_non_test, ]  # shuffle labels

    test_labels = loaddata(path + file_header + "test_label.csv")
    labels = np.concatenate((train_labels, test_labels), axis=0)

    nclass = np.amax(labels) + 1

    non_test_size = train_labels.shape[0]
    # val_size = int(non_test_size * val_ratio)
    # train_size = non_test_size - val_size
    total_size = features.shape[0]
    idx_train = range(non_test_size)
    idx_val = range(non_test_size, total_size)
    idx_test = range(non_test_size, total_size)

    if tensor_format:
        features = torch.FloatTensor(np.array(features))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return features, labels, idx_train, idx_val, idx_test, nclass


# load muse data in batch mode
def load_muse_bm(data_path, dataset, val_ratio=0.2):
    path = data_path + "muse/" + dataset + "/"
    file_header = dataset.lower() + "_"

    # load feature
    train_features = loadsparse(path + file_header + "train.csv")
    # shuttle train features
    non_test_size = train_features.shape[0]
    idx_non_test = random.sample(range(non_test_size), non_test_size)
    train_features = train_features[idx_non_test, ]

    test_features = loadsparse(path + file_header + "test.csv")
    features = sp.vstack([train_features, test_features])
    features = normalize(features)
    features = torch.FloatTensor(np.array(features))

    # load label
    train_labels = loaddata(path + file_header + "train_label.csv")
    train_labels = train_labels[idx_non_test, ]
    test_labels = loaddata(path + file_header + "test_label.csv")
    labels = np.concatenate((train_labels, test_labels), axis=0)
    label_dict = {i: l[0] for i, l in enumerate(labels.tolist())}
    nclass = np.amax(labels) + 1
    labels = torch.LongTensor(labels)


    # load partition: partition gathers the sample ID for training, validation and testing test.
    partition = {}
    non_test_size = train_labels.shape[0]
    val_size = int(non_test_size * val_ratio)
    train_size = non_test_size - val_size
    test_size = test_labels.shape[0]
    # idx_train, idx_val, idx_test = partition_data(train_size, val_size, test_size)
    partition["train"], partition["val"], partition["test"] = \
        range(non_test_size), range(train_size, non_test_size), range(non_test_size, non_test_size + test_size)

    # load labels

    # if tensor_format:
    #     features = torch.FloatTensor(np.array(features))
    #     labels = torch.LongTensor(labels)
    #
    #     idx_train = torch.LongTensor(idx_train)
    #     idx_val = torch.LongTensor(idx_val)
    #     idx_test = torch.LongTensor(idx_test)

    return features, labels, label_dict, partition, nclass


def normalize(mx, axis=1):
    """Row-normalize sparse matrix"""
    # rowsum = np.array(mx.sum(1))
    # r_inv = np.power(rowsum, -1).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    # mx = r_mat_inv.dot(mx)

    row_sums = mx.sum(axis)
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