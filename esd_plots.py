import weightwatcher as ww
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from operator import itemgetter
import os
import argparse
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
# from models import *
# from utils import progress_bar
import resnet_width_0
import resnet_width_1
import resnet_width_2
import resnet_width_3
import resnet_width_4
import resnet_width_5
import math
import warnings
import json
os.environ["OMP_NUM_THREADS"] = "1"
TRUNCATED_POWER_LAW = 'truncated_power_law'
XMIN_PEAK = 'xmin_peak'
E_TPL = 'E_TPL'
PL = 'PL'

import pandas as pd


net = resnet_width_4.ResNet34()
net.linear = nn.Linear(128, 100)

PATH = "/data/eliciaye/val_experiments/me-prune/cifar100/resnet34-1-0.075-evals/ckpt.pth"
checkpoint = torch.load(PATH)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['net'].items():
    if 'select' in k:
        v = v.view(1, -1, 1, 1)
    new_state_dict[k.replace('module.', '')] = v
net.load_state_dict(new_state_dict, strict=False)

watcher = ww.WeightWatcher(model=net)
res = dict()
plot_folder = "/data/eliciaye/val_experiments/me-prune/cifar100/resnet34-1-0.075-evals"
details = watcher.analyze(randomize=True, vectors=True, plot=True, savefig=plot_folder, fit='PL' or 'TPL' or 'E_TPL')
details.to_csv('resnet34-1-0.075.csv')
layers = details['layer_id'].values
# for l in layers:
#     res[l] = list(watcher.get_ESD(layer=l))


# PATH = "/rscratch/yyaoqing/alex/val_experiments/augprune-hydra/cifar100/resnet56-0.45-0.2-entropy-last/model_best.pth.tar"
# net = load_new(PATH)
# watcher = ww.WeightWatcher(model=net)
# res = dict()
# details = watcher.analyze()
# details.to_csv('pruned_details.csv')
# for l in layers:
#     res[l] = list(watcher.get_ESD(layer=l))
#
# with open('pruned_esd.json', 'w') as f:
#     json.dump(res, f)
