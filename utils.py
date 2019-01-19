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

def loadsparse2(fname):
    df = pd.read_csv(fname, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    row=np.max(a[:,0])
    column=np.max(a[:,1])
    s=sp.csr_matrix((a[:,2], (a[:,0],a[:,1])), shape=(row.astype('int64')+1, column.astype('int64')+1))
    return s

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
def load_muse_data(path="./data/", dataset="ECG", tensor_format=True, random_proj=True, random_projection_size=30000,random_projection_times=10):

    path = path + dataset + "/"
    file_header = dataset + "_"

    train_features = loadsparse2(path + file_header + "train.csv")
    test_features = loadsparse2(path + file_header + "test.csv")
    print(train_features.shape)
    print(test_features.shape)
    mf=np.min((test_features.shape[1],train_features.shape[1]));
    train_features=train_features[:,0:mf];
    test_features=test_features[:,0:mf];
    
    
    
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
        features = torch.FloatTensor(np.array(features.toarray()))
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
    mx=mx.astype('float32')
    row_sums_inverse=1/row_sums
    f=mx.multiply(row_sums_inverse)
    return sp.csr_matrix(f).astype('float32')

def convert2sparse(features):
    aaa=sp.coo_matrix(features)
    value = aaa.data
    column_index = aaa.col
    row_pointers = aaa.row
    a=np.array(column_index)
    b=np.array(row_pointers)
    a=np.reshape(a,(a.shape[0],1))
    b=np.reshape(b,(b.shape[0],1))
    s=np.concatenate((a,b),axis=1)
    t=torch.sparse.FloatTensor(torch.LongTensor(s.T),torch.FloatTensor(value))
    return t

def accuracy(output, labels):

    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

    return accuracy_score

def random_hash(features,K):
    idx=np.array(range(features.shape[1]));
    np.random.shuffle(idx)
    feat=features[:,idx]
    for i in range(features.shape[0]):
        f=np.array(feat[0].toarray())
        f.reshape


    tmp=torch.FloatTensor(features[:,idx[0:K]].toarray())
    return tmp

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


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
