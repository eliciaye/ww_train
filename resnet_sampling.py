'''Train CIFAR10 with PyTorch.'''
import weightwatcher as ww
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
import json
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
TRUNCATED_POWER_LAW = 'truncated_power_law'
XMIN_PEAK = 'xmin_peak'
E_TPL = 'E_TPL'
PL = 'PL'

warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
parser.add_argument('--Net', default='resnet_width_1.ResNet18()', help='neural network')
parser.add_argument('--checkpoint', default='res18cifar100_lr2load_width_1_D', help='pathway1')
parser.add_argument('--distribution', default='epoch_lr',choices=['epoch_lr', 'alpha','AWGN'], help='mean of samples')

parser.add_argument(
    '--resume',
    '-r',
    action='store_true',
    help='resume from checkpoint')
parser.add_argument(
    '--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument(
    '--seed',
    type=int,
    default=7,
    help='random seed')

args = parser.parse_args()
#writer = SummaryWriter('./tensorboard_logs/res18cifar100_lr2load/res18cifar100_lr2load_width_4_D_rectified_hard')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(
    root='/work/eliciaye', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='/work/eliciaye', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# Model
print('==> Building model..')

if args.Net == 'resnet_width_4.ResNet101()':
    net = resnet_width_4.ResNet101()
    net.linear= nn.Linear(128*4,100)
if args.Net == 'resnet_width_4.ResNet152()':
    net = resnet_width_4.ResNet152()
    net.linear= nn.Linear(128*4,100)
    
if args.Net == 'resnet_width_2.ResNet101()':
    net = resnet_width_2.ResNet101()
    net.linear= nn.Linear(256*4,100)
if args.Net == 'resnet_width_2.ResNet152()':
    net = resnet_width_2.ResNet152()
    net.linear= nn.Linear(256*4,100)

if args.Net=='resnet_width_0.ResNet18()':
    net = resnet_width_0.ResNet18()
    net.linear= nn.Linear(512,100)

if args.Net=='resnet_width_1.ResNet18()':
    net = resnet_width_1.ResNet18()
    net.linear= nn.Linear(1024,100)
if args.Net=='resnet_width_2.ResNet18()':
    net = resnet_width_2.ResNet18()
    net.linear= nn.Linear(256,100)

if args.Net=='resnet_width_3.ResNet18()':
    net = resnet_width_3.ResNet18()
    net.linear = nn.Linear(768, 100)
if args.Net=='resnet_width_4.ResNet18()':
    net = resnet_width_4.ResNet18()
    net.linear = nn.Linear(128, 100)
if args.Net=='resnet_width_5.ResNet18()':
    net = resnet_width_5.ResNet18()
    net.linear = nn.Linear(1280, 100)

if args.Net=='resnet_width_0.ResNet34()':
    net = resnet_width_0.ResNet34()
    net.linear= nn.Linear(512,100)
if args.Net=='resnet_width_1.ResNet34()':
    net = resnet_width_1.ResNet34()
    net.linear= nn.Linear(1024,100)
if args.Net=='resnet_width_2.ResNet34()':
    net = resnet_width_2.ResNet34()
    net.linear= nn.Linear(256,100)

if args.Net=='resnet_width_3.ResNet34()':
    net = resnet_width_3.ResNet34()
    net.linear = nn.Linear(768, 100)
if args.Net=='resnet_width_4.ResNet34()':
    net = resnet_width_4.ResNet34()
    net.linear = nn.Linear(128, 100)
if args.Net=='resnet_width_5.ResNet34()':
    net = resnet_width_5.ResNet34()
    net.linear = nn.Linear(1280, 100)

print("Model Name: ", args.Net)
print(net)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    # assert os.path.isdir(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    trainacc = 100. * correct / total
    wwtrainacc.append(trainacc)
    print("Finished training Epoch {}, Train Acc = {}\nTrain loss = {}".format(epoch, trainacc,train_loss))
    #writer.add_scalar('train_acc', trainacc, global_step=epoch)
   # writer.add_scalar('train_loss', train_loss, global_step=epoch)


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    testacc = 100.*correct/total

    print("Finished evaluating Epoch {}. Test Acc = {}\nTest loss={}\n".format(epoch,testacc,test_loss))
   # writer.add_scalar('test_acc', testacc, global_step=epoch)
    #writer.add_scalar('test_loss', test_loss, global_step=epoch)
    return testacc

wwtrainacc=[]
wwtestacc=[]
wwgeneration_gap=[]
recalpha=[]
reclr=[]
allalpha=[]
best_acc = 0.
watcher = ww.WeightWatcher(model=net)
details = watcher.analyze(vectors=False, fix_fingers=False, fit=PL)
for b in range(details.shape[0]):
    reclr.append([])
    allalpha.append([])
for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    
    watcher = ww.WeightWatcher(model=net)
    details = watcher.analyze(vectors=False,fix_fingers=False,fit=PL)
    n = details.shape[0]
    summary = watcher.get_summary(details)
    details_path = os.path.join(args.checkpoint, 'details.csv')
    details.to_csv(details_path)
    for i in range(details.shape[0]):
        allalpha[i].append(details.loc[i,'alpha'])
    recalpha.append(summary['alpha'])
    #writer.add_scalars(main_tag='alpha/layer',
                      # tag_scalar_dict={str(i): details.loc[i, 'alpha'] for i in
                      #                  range(details.shape[0])},
                      # global_step=epoch)
    para_save=[]
    net_para1 = []

    for name,para in net.named_parameters():
        if ('conv' in name or 'shortcut.0' in name or 'linear' in name) and ('weight' in name):
            net_para1.append(para)

    alpha_all=np.array([details.loc[i, 'alpha'] for i in range(n)])
    sorted_alpha_idx = np.argsort(alpha_all)
    l = args.lr * (1 + math.cos((epoch + 1) * math.pi / args.epochs)) / 2

    if args.distribution == 'epoch_lr':
        samples = np.sort(np.random.normal(l,1,n))
        # shift to make nonnegative
        samples = samples - min(samples)
        samples_prob = np.divide(samples,np.sum(samples))

        for i in range(n):
            para_save.append({'params': net_para1[sorted_alpha_idx[i]],
                            'lr': l * n * samples_prob[i], 'momentum': 0.9,
                            'weight_decay': 5e-4})
    elif args.distribution == 'alpha':
        samples = []
        for i in range(n):
            samples.append(np.random.normal(alpha_all[i],1,1).item())
        samples=np.array(samples)
        samples = samples - min(samples)
        samples_prob = np.divide(samples,np.sum(samples))
        for i in range(n):
            para_save.append({'params': net_para1[i],
                            'lr': l * n * samples_prob[i], 'momentum': 0.9,
                            'weight_decay': 5e-4})
    elif args.distribution == 'AWGN':
        samples = np.sort(np.random.normal(np.mean(alpha_all),np.sqrt(1+np.std(alpha_all)),n))
        # shift to make nonnegative
        samples = samples - min(samples)
        samples_prob = np.divide(samples,np.sum(samples))

        for i in range(n):
            para_save.append({'params': net_para1[sorted_alpha_idx[i]],
                            'lr': l * n * samples_prob[i], 'momentum': 0.9,
                            'weight_decay': 5e-4})
    all_params = [value for index, value in sorted(list(zip(np.argsort(alpha_all), para_save)), key=itemgetter(0))]

    params_id = list(map(id, net_para1))
    other_params = list(filter(lambda p: id(p) not in params_id, net.parameters()))
    all_params.append({'params': other_params})
    optimizer = torch.optim.SGD(all_params, lr=l, momentum=0.9, weight_decay=5e-4)

    for m in range(n):
        reclr[m].append(optimizer.state_dict()['param_groups'][m]['lr'])
    print([reclr[m][-1] for m in range(n)])

    testacc=test(epoch)
    wwtestacc.append(testacc)
    if testacc > best_acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': testacc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.checkpoint):
            os.mkdir(args.checkpoint)
        torch.save(state, args.checkpoint + '/ckpt.pth')
        best_acc = testacc

wwtrainacc_1 = deepcopy(wwtrainacc)
wwtestacc_1 = deepcopy(wwtestacc)
wwtrainacc_1.sort()
wwtrainacc_1.reverse()
wwtestacc_1.sort()
wwtestacc_1.reverse()
wwtrainacc_final = (wwtrainacc_1[0] + wwtrainacc_1[1] + wwtrainacc_1[2] + wwtrainacc_1[3] + wwtrainacc_1[4]) / 5
wwtestacc_final = (wwtestacc_1[0] + wwtestacc_1[1] + wwtestacc_1[2] + wwtestacc_1[3] + wwtestacc_1[4]) / 5

print("Sampling {}: Epoch {}, Final Test Acc = {}".format(args.distribution, args.epochs,wwtestacc_final))
print("Sampling {}: Epoch {}, Final Train Acc = {}".format(args.distribution, args.epochs,wwtrainacc_final))

summary_path = os.path.join(args.checkpoint,'training_log.txt')
with open(summary_path,'w+') as f:
    f.write('trainacc='+str(wwtrainacc)+'\n')
    f.write('testacc'+str(wwtestacc)+'\n')
    f.write('averagealpha='+str(recalpha)+'\n')
    f.write(str(wwtrainacc_final) + '\n')
    f.write(str(wwtestacc_final) + '\n')
    for m in range(details.shape[0]):
        f.write('lr_layer{}='.format(m+1)+str(reclr[m])+'\n')
    for m in range(details.shape[0]):
        f.write('alpha_layer{}='.format(m+1)+str(allalpha[m])+'\n')

    f.close()
