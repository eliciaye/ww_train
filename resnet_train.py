import weightwatcher as ww
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from operator import itemgetter
import os
import math
import numpy as np
import json
import warnings
import argparse
from copy import deepcopy
from training import train, test

import resnet_widths_all
import resnet_width_4
import data
from utils import *

os.environ["OMP_NUM_THREADS"] = "1"
TRUNCATED_POWER_LAW = 'truncated_power_law'
XMIN_PEAK = 'xmin_peak'
E_TPL = 'E_TPL'
PL = 'PL'

warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset')
parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--checkpoint', default='/data/eliciaye/ww_train/res18-25_2', help='ckpt dir path')
parser.add_argument('--datadir', default='/work/eliciaye/', help='directory of dataset')
parser.add_argument('--width_frac', default=1.0,type=float, help='fraction of original width')
parser.add_argument(
    '--sample_evals',
    '-r',
    action='store_true',
    help='sample eigenvalues and replace during ww SVD analysis')
parser.add_argument(
    '--lr_rewind',
    action='store_true',
    help='learning rate rewinding')
parser.add_argument(
    '--resume',
    '-r',
    action='store_true',
    help='resume from checkpoint')
parser.add_argument(
    '--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument(
    '--depth', type=int, default=18, help='Number of layers in model')
parser.add_argument(
    '--seed',
    type=int,
    default=7,
    help='random seed')
parser.add_argument(
    '--temp_balance_lr',
    type=str,
    default='',
    help='use tempbalance for learning rate')
parser.add_argument(
    '--temp_balance_wd',
    type=str,
    default='',
    help='use tempbalance for weight decay')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
trainloader, testloader=data.get_data_loaders(args.datadir)

# Model
print('==> Building model..')

if args.depth == 18:
    net = resnet_widths_all.ResNet18(dataset=args.dataset,width_frac=args.width_frac)
elif args.depth == 34:
    net = resnet_widths_all.ResNet34(dataset=args.dataset,width_frac=args.width_frac)
elif args.depth == 50:
    net = resnet_widths_all.ResNet50(dataset=args.dataset,width_frac=args.width_frac)
elif args.depth == 101:
    net = resnet_widths_all.ResNet101(dataset=args.dataset,width_frac=args.width_frac)
elif args.depth == 152:
    net = resnet_widths_all.ResNet152(dataset=args.dataset,width_frac=args.width_frac)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

start_epoch=0
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint+'/ckpt.pth')
    net.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=args.wd)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

epoch_trainaccs=[]
epoch_testaccs=[]

best_acc = 0.

watcher = ww.WeightWatcher(model=net)
details = watcher.analyze(vectors=False, fix_fingers=False, fit=PL)
n = details.shape[0]
epoch_layer_lrs = [[] for _ in range(n)]
epoch_layer_alphas = [[] for _ in range(n)]

cosine_anneal_lr_timeline = [args.lr * (1 + math.cos((epoch + 1) * math.pi / args.epochs)) / 2 for epoch in range(args.epochs)]
layerwise_lr_timeline_idx = np.repeat(0, len(param_groups))

prev_epoch_alphas = []

for epoch in range(start_epoch, start_epoch+args.epochs):
    trainacc, train_loss = train(epoch, net, trainloader, device, optimizer, criterion)
    epoch_trainaccs.append(trainacc)
    print("Finished training Epoch {}. Train Acc = {}\nTrain loss = {}".format(epoch,trainacc,train_loss))
    
    testacc, test_loss = test(epoch, net, testloader, device, criterion)
    epoch_testaccs.append(testacc)
    print("Finished evaluating Epoch {}. Test Acc = {}\nTest loss={}\n".format(epoch,testacc,test_loss))

    watcher = ww.WeightWatcher(model=net)
    details = watcher.analyze(vectors=False, plot=True, savefig=args.checkpoint+'/before_train', fix_fingers=False, fit=PL, sample_evals=args.sample_evals)

    details_path = os.path.join(args.checkpoint, 'details.csv')
    details.to_csv(details_path)

    n = details.shape[0]
    for i in range(n):
        epoch_layer_alphas[i].append(details.loc[i,'alpha'])

    ww_params = []
    other_params = []
    for name,para in net.named_parameters():
        if ('conv' in name or 'shortcut.0' in name or 'linear' in name) and ('weight' in name):
            ww_params.append(para)
        else:
            other_params.append(para)

    n_alphas=[details.loc[i, 'alpha'] for i in range(n)]
    print(len(n_alphas),"alphas: ",n_alphas)

    epoch_lr = args.lr * (1 + math.cos((epoch + 1) * math.pi / args.epochs)) / 2
    epoch_wd = args.wd # change if use weight decay schedule

    all_params=[]
    lrs = get_layer_temps(args.temp_balance_lr,n_alphas,epoch_lr)
    if epoch > 0 and args.lr_rewind:
        alpha_ratios = np.divide(n_alphas,prev_epoch_alphas)
        layerwise_lr_timeline_idx[alpha_ratios>1.5] = min(0,epoch-20)
        lrs = [cosine_anneal_lr_timeline[layerwise_lr_timeline_idx[i]] for i in range(n)]
        prev_epoch_alphas=n_alphas
    for i in range(n):
        epoch_layer_lrs[i].append(lrs[i])
    print("Layerwise learning rates for epoch {}:".format(epoch),lrs)
    
    wds = get_layer_temps(args.temp_balance_wd,n_alphas,epoch_wd) 
    
    for i in range(n):
        all_params.append({'params': ww_params[i],
                        'lr': lrs[i], 'momentum': 0.9,
                        'weight_decay': wds[i]})

    all_params.append({'params': other_params})
    optimizer = torch.optim.SGD(all_params, lr=epoch_lr, momentum=0.9, weight_decay=epoch_wd)

    if testacc > best_acc:
        print('Saving...')
        state = {
            'state_dict': net.state_dict(),
            'best_acc': testacc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.checkpoint):
            os.mkdir(args.checkpoint)
        torch.save(state, args.checkpoint + '/ckpt.pth')
        best_acc = testacc

TOP_VALS = 5
epoch_trainaccs.sort(reverse=True)
epoch_testaccs.sort(reverse=True)
epoch_trainaccs_final = (epoch_trainaccs[:TOP_VALS]) / TOP_VALS
epoch_testaccs_final = (epoch_testaccs[:TOP_VALS]) / TOP_VALS

print("Epoch {}, Test Acc = {}\nFinal Train Acc = {}\n".format(args.epochs,epoch_testaccs_final,epoch_trainaccs_final))

summary_path = os.path.join(args.checkpoint,'training_log.txt')
with open(summary_path,'w+') as f:
    f.write('epoch_trainaccs='+str(epoch_trainaccs)+'\n')
    f.write('epoch_testaccs='+str(epoch_testaccs)+'\n')
    f.write('Final Train Accuracy='+str(epoch_trainaccs_final) + '\n')
    f.write('Final Test Accuracy='+str(epoch_testaccs_final) + '\n')
    for m in range(len(epoch_layer_lrs)):
        f.write('lr_layer{}='.format(m+1)+str(epoch_layer_lrs[m])+'\n')
    for m in range(len(epoch_layer_alphas)):
        f.write('alpha_layer{}='.format(m+1)+str(epoch_layer_alphas[m])+'\n')

    f.close()