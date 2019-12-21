import torch
from torch import nn
import numpy as np
from torch.autograd import Variable


# x = torch.zeros(10)
# x[0] = 0
# x = x.view(-1,1).repeat(1, 6).t()
# y = torch.ones(6,10)
# f_loss = nn.CosineSimilarity(eps=1e-9)
# print(f_loss(x, y))
# print(x, y)

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize=2):
        self.cuda = cuda;
        self.P = window;
        self.h = horizon
        fin = open(file_name);
        self.rawdat = np.loadtxt(fin, delimiter=',');
        self.dat = np.zeros(self.rawdat.shape);
        self.n, self.m = self.dat.shape;
        self.normalize = 2
        self.scale = np.ones(self.m);
        self._normalized(normalize);
        self._split(int(train * self.n), int((train + valid) * self.n), self.n);

        self.scale = torch.from_numpy(self.scale).float();
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m);

        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);

        #self.rse = normal_std(tmp);
        #self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat);

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]));
                self.dat[:, i] = 2 * ( (self.rawdat[:, i] - np.min(np.abs(self.rawdat[:, i]))) / (np.max(np.abs(self.rawdat[:, i]))- np.min(np.abs(self.rawdat[:, i]))) ) - 1;

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train);
        valid_set = range(train, valid);
        test_set = range(valid, self.n);
        self.train = self._batchify(train_set, self.h);
        self.valid = self._batchify(valid_set, self.h);
        self.test = self._batchify(test_set, self.h);

    def _batchify(self, idx_set, horizon):

        n = len(idx_set);
        # X = torch.zeros((n, self.P, self.m));
        X = torch.zeros((n * self.m, 1, self.P));
        # Y = torch.zeros((n, self.m));
        Y = torch.zeros((n*self.m, 1));

        print('begin')
        for i in range(n):
            # print(i)
            end = idx_set[i] - self.h + 1;
            start = end - self.P;
            # X[i, :, :] = torch.from_numpy(self.dat[start:end, :]);
            # Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :]);
            for j in range(self.m):
                X[i+j, 0, :] = torch.from_numpy(self.dat[start:end, j])
                Y[i+j][0] = self.dat[idx_set[i], j]

        print('end')
        return [X, Y];


Data = Data_utility('./electricity.txt', 0.6, 0.2, True, 12, 7*24, 2)

print(Data.train[0].shape)
print(Data.train[1].shape)
train_x_npy = Data.train[0].numpy()
train_y_npy = Data.train[1].numpy()
test_x_npy = Data.test[0].numpy()
test_y_npy = Data.test[1].numpy()

path = "./dataset/" + 'electricity'+ "/"
np.save(path + 'X_test.npy', test_x_npy)
np.save(path + 'X_train.npy', train_x_npy)
np.save(path + 'y_test.npy', test_y_npy)
np.save(path + 'y_train.npy', train_y_npy)