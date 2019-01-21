from __future__ import division
from __future__ import print_function

import time
import argparse

import torch.optim as optim

from utils import *
from dp_models import *
from models import *
from torch.utils import data
from bm.ts_dataset import TSDataset

datasets = ["ArticularyWordRecognition", "AtrialFibrilation", "BasicMotions", "CharacterTrajectories", "Cricket",
            "EigenWorms", "Epilepsy", "ERing", "EthanolConcentration", "FingerMovements",
             "HandMovementDirection", "Handwriting", "Heartbeat", "JapaneseVowels", "Libras",
            "LSST", "MotorImagery", "NATOPS", "PEMS-SF", "PenDigits",
            "Phoneme", "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits",
            "StandWalkJump", "UWaveGestureLibrary", "", "", ""]

# parameter settings
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="./data/",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="ECG",
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=3000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.00005,
                    help='Initial learning rate. default:[0.00005]')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability). Default:0.5')
args = parser.parse_args()

# configure cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# init dataset
# Parameters
params = {'batch_size': 100,
          'shuffle': False,
          'num_workers': 1}

# Datasets
features, labels, label_dict, partition, nclass = load_muse(args.data_path, args.dataset)

# Generators
training_set = TSDataset(features, partition['train'], label_dict)
training_generator = data.DataLoader(training_set, **params)

validation_set = TSDataset(features, partition['val'], label_dict)
validation_generator = data.DataLoader(validation_set, **params)

test_set = TSDataset(features, partition['test'], label_dict)
test_generator = data.DataLoader(test_set, **params)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
print("Loading dataset", args.dataset, "...")
# Model and optimizer
model_type = "ProtoGCN"  # Options: FGCN, SemiProtoGCN, ProtoGCN, BiGCN, MotifGCN, InterGCN, TPNet
if model_type == "ProtoGCN":
    # features, labels, idx_train, idx_val, idx_test, nclass = load_muse_data(args.data_path + "muse/", dataset=args.dataset)
    model = TapNet(nfeat=features.shape[1],
                   nhid=args.hidden,
                   nclass=nclass,
                   dropout=args.dropout)
    #input = (features, labels, idx_train)
elif model_type == "SemiProtoGCN":
    features, labels, idx_train, idx_val, idx_test, nclass = load_muse_data(args.data_path + "muse/", dataset=args.dataset)
    model = SemiProtoGCN(nfeat=features.shape[1],
                         nhid=args.hidden,
                         nclass=nclass,
                         dropout=args.dropout)
    # cuda
    if args.cuda:
        model.cuda()
        features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()
    input = (features, labels, idx_train, idx_val, idx_test)
elif model_type == "TPNet":
    ts, labels, idx_train, idx_val, idx_test, nclass = load_raw_ts(args.data_path + "raw/", dataset=args.dataset)
    model = TPNet(channel=ts.size(1),
                  ts_length=ts.size(2),
                  nclass=nclass,
                  dropout=args.dropout)
    if args.cuda:
        model.cuda()
        ts, labels, idx_train = ts.cuda(), labels.cuda(), idx_train.cuda()
    input = (ts, labels, idx_train)

# init the optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


# training function
def train():

    # Loop over epochs
    for epoch in range(args.epochs):
        # Training
        t = time.time()
        model.train()
        optimizer.zero_grad()
        acc_train_list, bs_train_list, loss_train_list, prototype_list = [], [], [], []
        for batch_feature, batch_labels in training_generator:
            # Transfer to GPU
            if args.cuda:
                batch_feature, batch_labels = batch_feature.cuda(), batch_labels.cuda()

            ts_proto, output = model(batch_feature, batch_labels)
            loss_train = F.cross_entropy(output, torch.squeeze(batch_labels))
            acc_train = accuracy(output, batch_labels)
            loss_train.backward()
            optimizer.step()
            acc_train_list.append(acc_train)
            bs_train_list.append(output.shape[0])
            loss_train_list.append(loss_train)
            prototype_list.append(ts_proto)
        acc_train = sum([acc_i * bs_train_list[i] for i, acc_i in enumerate(acc_train_list)]) / sum(bs_train_list)
        loss_train = sum(loss_train_list)

        # evaluate validation set
        acc_val = test(validation_generator, ts_proto)
        acc_test = test(test_generator, ts_proto)
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'acc_test: {:.4f}'.format(acc_test.item()),
              'time: {:.4f}s'.format(time.time() - t))

# test function
def test(generator, ts_proto):

    acc_list, bs_list = [], []
    for batch_feature, batch_labels in generator:
        # Transfer to GPU
        if args.cuda:
            batch_feature, batch_labels = batch_feature.cuda(), batch_labels.cuda()

        _, output = model(batch_feature, batch_labels, ts_proto=ts_proto)
        # loss_val = F.cross_entropy(output, torch.squeeze(batch_labels))
        acc = accuracy(output, batch_labels)
        acc_list.append(acc)
        bs_list.append(output.shape[0])
    acc = sum([acc_i * bs_list[i] for i, acc_i in enumerate(acc_list)]) / sum(bs_list)
    return acc

# Train model
t_total = time.time()
train()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
