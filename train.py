from __future__ import division
from __future__ import print_function

import math
import sys
import time
import argparse

import torch.optim as optim
from torch import nn
from models import TapNet
from utils import *
import torch.nn.functional as F

datasets = ["ArticularyWordRecognition", "AtrialFibrilation", "BasicMotions", "CharacterTrajectories", "Cricket",
            "EigenWorms", "Epilepsy", "ERing", "EthanolConcentration", "FingerMovements",
             "HandMovementDirection", "Handwriting", "Heartbeat", "JapaneseVowels", "Libras",
            "LSST", "MotorImagery", "NATOPS", "PEMS-SF", "PenDigits",
            "Phoneme", "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits",
            "StandWalkJump", "UWaveGestureLibrary", "", "", ""]

parser = argparse.ArgumentParser()

# dataset settings
parser.add_argument('--data_path', type=str, default="./dataset/",
                    help='the path of data.')
parser.add_argument('--use_muse', action='store_true', default=False,
                    help='whether to use the raw data. Default:False')
parser.add_argument('--dataset', type=str, default="NATOPS", #NATOPS
                    help='time series dataset. Options: See the datasets list')

# cuda settings
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# Training parameter settings
parser.add_argument('--epochs', type=int, default=3000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='Initial learning rate. default:[0.00001]')
parser.add_argument('--wd', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters). default: 5e-3')
parser.add_argument('--stop_thres', type=float, default=1e-9,
                    help='The stop threshold for the training error. If the difference between training losses '
                         'between epoches are less than the threshold, the training will be stopped. Default:1e-9')

# Model parameters
parser.add_argument('--use_lstm', type=boolean_string, default=True,
                    help='whether to use LSTM for feature extraction. Default:False')
parser.add_argument('--use_cnn', type=boolean_string, default=True,
                    help='whether to use CNN for feature extraction. Default:False')
parser.add_argument('--use_rp', type=boolean_string, default=False,
                    help='Whether to use random projection')
parser.add_argument('--rp_params', type=str, default='-1,3',
                    help='Parameters for random projection: number of random projection, '
                         'sub-dimension for each random projection')
parser.add_argument('--use_metric', action='store_true', default=False,
                    help='whether to use the metric learning for class representation. Default:False')
parser.add_argument('--metric_param', type=float, default=0.000001,
                    help='Metric parameter for prototype distances between classes. Default:0.000001')
parser.add_argument('--use_ss', action='store_true', default=False,
                    help='Use semi-supervised learning.')
parser.add_argument('--filters', type=str, default="256,256,128",
                    help='filters used for convolutional network. Default:256,256,128')
parser.add_argument('--kernels', type=str, default="8,5,3",
                    help='kernels used for convolutional network. Default:8,5,3')
parser.add_argument('--dilation', type=int, default=1,
                    help='the dilation used for the first convolutional layer. '
                         'If set to -1, use the automatic number. Default:-1')
parser.add_argument('--layers', type=str, default="500,300",
                    help='layer settings of mapping function. [Default]: 500,300')
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
args.kernels = [int(l) for l in args.kernels.split(",")]
args.filters = [int(l) for l in args.filters.split(",")]
args.rp_params = [int(l) for l in args.rp_params.split(",")]

if not args.use_lstm and not args.use_cnn:
    print("Must specify one encoding method: --use_lstm or --use_cnn")
    print("Program Exiting.")
    exit(-1)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
print("Loading dataset", args.dataset, "...")
# Model and optimizer
model_type = "TapNet"  # Options: FGCN, ProtoGCN, BiGCN, MotifGCN, InterGCN, TPNet, TapNet
if model_type == "TapNet":

    if args.use_muse:
        features, labels, idx_train, idx_val, idx_test, nclass \
                                    = load_muse(args.data_path, dataset=args.dataset, sparse=args.sparse)
    else:
        features, labels, idx_train, idx_val, idx_test, nclass \
                                    = load_raw_ts(args.data_path, dataset=args.dataset)

    #features, labels, idx_train, idx_val, idx_test, nclass = load_muse(args.data_path, dataset=args.dataset, sparse=True)

    # update random permutation parameter
    if args.rp_params[0] < 0:
        # dim = features.shape[1]
        # if dim <= 6:
        #     args.rp_params = [dim, math.ceil(dim / 2)]
        # elif dim > 6 and dim <= 20:
        #     args.rp_params = [10, 3]
        # else:
        #     args.rp_params = [int(dim / 2), 3]
        dim = features.shape[1]
        args.rp_params = [3, math.floor(dim * 2 / 3)]

    print("rp_params:", args.rp_params)

    # update dilation parameter
    if args.dilation == -1:
        args.dilation = math.floor(features.shape[2] / 64)

    print("Data shape:", features.size())
    model = TapNet(nfeat=features.shape[1],
                   len_ts=features.shape[2],
                   layers=args.layers,
                   nclass=nclass,
                   dropout=args.dropout,
                   use_lstm=args.use_lstm,
                   use_cnn=args.use_cnn,
                   filters=args.filters,
                   dilation=args.dilation,
                   kernels=args.kernels,
                   use_ss=args.use_ss,
                   use_metric=args.use_metric,
                   use_rp=args.use_rp,
                   rp_params=args.rp_params
                   )

    # cuda
    if args.cuda:
        model.cuda()
        labels, idx_train = labels.cuda(), idx_train.cuda()
    input = (features, labels, idx_train, idx_val, idx_test)

# init the optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.wd)


# training function
device = torch.device("cuda")
def train():
    loss_list = [sys.maxsize]
    test_best_mae, best_so_far = sys.maxsize, sys.maxsize
    step = int(idx_train.shape[0] / 1000)
    for epoch in range(args.epochs):
        start = 0
        for i in range(1000):
            if i == 999:
                batch_idx = range(start, idx_train.shape[0])
            else:
                batch_idx = range(start, start + step)
            start += step
            t = time.time()
            model.train()
            optimizer.zero_grad()

            batch_input = features[batch_idx]
            batch_input = batch_input.cuda()
            output, upp = model(batch_input)
            cos = nn.CosineSimilarity(eps=1e-9)

            # x 360*300 upp 300*360
            loss_train = (labels[batch_idx] - cos(output, upp)).pow(2).sum()
            # mae = accuracy(output[idx_train], labels[idx_train])
            scores_train = (cos(output, upp) + 1) / 2.
            # print(scores_train[0], labels[0])
            pred_train = (scores_train * (76400. - 0.)) + 0.
            true_labels = (labels[batch_idx] + 1) / 2 * (76400. - 0.)
            true_labels = true_labels.view(-1)
            # print(pred_train[0], true_labels[0])
            mae_train = torch.abs(pred_train - true_labels).sum()
            print('Epoch: {:04d}'.format(epoch + 1),
                    'iteration: {:04d}'.format(i+1),
                    'loss_train: {:.8f}'.format(loss_train.item()),
                    'mae_train:{:.4f}'.format(mae_train.item()/len(batch_idx)))
            loss_train.backward()
            optimizer.step()
                # output_val, upp_val = model(features[idx_val])
                # loss_val = (labels[idx_val] - cos(output_val, upp_val)).pow(2).sum()
                # scores_val = (cos(output_val, upp_val) + 1) / 2.
                # mae_val = torch.abs(scores_val - labels[idx_val]).sum()
                # # print(output[idx_val])
                # print('Epoch: {:04d}'.format(epoch + 1),
                      # 'loss_train: {:.8f}'.format(loss_train.item()),
                      # 'mae_train: {:.4f}'.format(mae_train.item()),
                      # 'loss_val: {:.4f}'.format(loss_val.item()),
                      # 'mae_val: {:.4f}'.format(mae_val.item()/len(val_idx)),
                      # 'time: {:.4f}s'.format(time.time() - t))

                # if mae_val.item() / len (idx_val) < test_best_mae:
                    # test_best_mae = mae_val.item()
                # if best_so_far > loss_train.item():
                    # best_so_far = loss_train.item()
                    # test_mae = mae_val.item()/len(idx_val)

        #new_input = (features[idx_train, ], labels[idx_train], idx_train, idx_val, idx_test)
        # loss_train, output = model(input)
        # print(features[idx_train])
        # print(output[idx_train])

        # loss_train = F.cross_entropy(output[idx_train], torch.squeeze(labels[idx_train]))
        # if args.use_metric:
        #     loss_train = loss_train - args.metric_param * proto_dist
        #
        # if abs(loss_train.item() - loss_list[-1]) < args.stop_thres \
        #         or loss_train.item() > loss_list[-1]:
        #     break
        # else:
        #     loss_list.append(loss_train.item())


        # if not args.fastmode:
        #     # Evaluate validation set performance separately,
        #     # deactivates dropout during validation run.
        #     model.eval()
        #     output = model(features)

        #print(output[idx_val])
        # loss_val = F.cross_entropy(output[idx_val], torch.squeeze(labels[idx_val]))
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        # loss_val = (labels[idx_val] - cos(output[idx_val], upp[idx_val])).pow(2).sum()
        # scores_val = (cos(output[idx_val], upp[idx_val]) + 1) / 2.
        # mae_val = torch.abs(scores_val[idx_val] - labels[idx_val]).sum()
        # # print(output[idx_val])
        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.8f}'.format(loss_train.item()),
        #       'mae_train: {:.4f}'.format(mae_train.item()),
        #       'loss_val: {:.4f}'.format(loss_val.item()),
        #       'mae_val: {:.4f}'.format(mae_val.item()),
        #       'time: {:.4f}s'.format(time.time() - t))

        # if mae_val.item() < test_best_mae:
        #     test_best_mae = mae_val.item()
        # if best_so_far > loss_train.item():
        #     best_so_far = loss_train.item()
        #     test_mae = mae_val.item()
    # print("test_acc: " + str(test_mae))
    # print("best possible: " + str(test_best_mae))

# test function
def test():
    # output, proto_dist = model(input)
    step = int(idx_test.shape[0] / 1000)
    input_test = features[idx_test]
    start = 0
    mae_test = 0
    for i in range(1000):
        if i == 999:
            batch_idx = range(start, idx_test.shape[0])
        else:
            batch_idx = range(start, start + step)
        start += step

        batch_input = features[batch_idx]
        batch_input = batch_input.cuda()
        output, upp = model(batch_input)
        cos = nn.CosineSimilarity(eps=1e-9)
        # print(output[idx_test])
        loss_test = (labels[batch_idx] - cos(output, upp)).pow(2).sum()
        scores_test = (cos(output, upp) + 1) / 2.
        pred_test = (scores_test * (764000. - 0.)) + 0.
        pred_test = pred_test.view(-1)
        true_labels_test = labels[batch_idx]
        true_labels_test = true_labels_test.view(-1)
        mae_test += torch.abs(pred_test - true_labels_test).sum().item()
    # loss_test = F.cross_entropy(output[idx_test], torch.squeeze(labels[idx_test]))
    # if args.use_metric:
    #     loss_test = loss_test - args.metric_param * proto_dist
    #
    # acc_test = accuracy(output[idx_test], labels[idx_test])
    print(args.dataset, "Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "mae= {:.4f}".format(mae_test/len(idx_test)))

# Train model
t_total = time.time()
train()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
