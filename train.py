from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import *

datasets = ["ArticularyWordRecognition", "AtrialFibrilation", "BasicMotions", "CharacterTrajectories", "Cricket",
            "EigenWorms", "Epilepsy", "ERing", "EthanolConcentration", "FingerMovements",
             "HandMovementDirection", "Handwriting", "Heartbeat", "JapaneseVowels", "Libras",
            "LSST", "MotorImagery", "NATOPS", "PEMS-SF", "PenDigits",
            "Phoneme", "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits",
            "StandWalkJump", "UWaveGestureLibrary", "", "", ""]
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="./data/",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="NATOPS",
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--ss', action='store_true', default=False,
                    help='Use semi-supervised learning.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.00005,
                    help='Initial learning rate. default:[0.00005]')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters). default: 5e-3')
parser.add_argument('--layers', type=str, default="2000,1000,300",
                    help='layer settings of mapping function.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability). Default:0.5')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
args.sparse = True
args.layers = [int(l) for l in args.layers.split(",")]
# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
print("Loading dataset", args.dataset, "...")
# Model and optimizer
model_type = "TapNet"  # Options: FGCN, ProtoGCN, BiGCN, MotifGCN, InterGCN, TPNet, TapNet
if model_type == "TapNet":
    if args.dataset=="ECG":
        args.sparse = False

    features, labels, idx_train, idx_val, idx_test, nclass = load_muse(args.data_path, dataset=args.dataset, sparse=args.sparse)
    #features, labels, idx_train, idx_val, idx_test, nclass = load_muse(args.data_path, dataset=args.dataset, sparse=True)
    model = TapNet(nfeat=features.shape[1],
                   layers=args.layers,
                   nclass=nclass,
                   dropout=args.dropout,
                   use_ss=False)
    # cuda
    if args.cuda:
        model.cuda()
        features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()
    input = (features, labels, idx_train, idx_val, idx_test)

# init the optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


# training function
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(input)
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

    #print(output[idx_val])
    loss_val = F.cross_entropy(output[idx_val], torch.squeeze(labels[idx_val]))
    acc_val = accuracy(output[idx_val], labels[idx_val])
    # print(output[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.8f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


# test function
def test():
    output = model(input)
    #print(output[idx_test])
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
