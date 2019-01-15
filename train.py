from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import FGCN, ProtoGCN, MotifGCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-1,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability). Default:0.5')
parser.add_argument('--dataset', type=str, default="ECG",
                    help='time series dataset. Options: ECG, PEN')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
print("Loading dataset", args.dataset, "...")
features, labels, idx_train, idx_val, idx_test, nclass = load_muse_data(dataset=args.dataset)

# Model and optimizer
model_type = "ProtoGCN"  # Options: FGCN, ProtoGCN, MotifGCN

if model_type == "FGCN":
    model = FGCN(nfeat=features.shape[1],
                 nhid=args.hidden,
                 nclass=nclass,
                 dropout=args.dropout)
elif model_type == "ProtoGCN":
    model = ProtoGCN(nfeat=features.shape[1],
                     nhid=args.hidden,
                     nclass=nclass,
                     dropout=args.dropout)
elif model_type == "MotifGCN":
    model = MotifGCN(ts_feat=200,
                     motif_feat=100,
                     nhid=args.hidden,
                     nclass=nclass,
                     dropout=args.dropout)


optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, labels, idx_train)
    # print(features[idx_train])
    #print(output[idx_train])

    loss_train = F.cross_entropy(output[idx_train], torch.squeeze(labels[idx_train]))
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features)

    model.eval()
    loss_val = F.cross_entropy(output[idx_val], torch.squeeze(labels[idx_val]))
    acc_val = accuracy(output[idx_val], labels[idx_val])
    #print(output[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, labels, idx_train)
    print(output[idx_val])
    loss_test = F.cross_entropy(output[idx_val], torch.squeeze(labels[idx_val]))
    acc_test = accuracy(output[idx_val], labels[idx_val])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    model.eval()
    output = model(features, labels, idx_train)
    print(output[idx_test])
    loss_test = F.cross_entropy(output[idx_test], torch.squeeze(labels[idx_test]))
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
