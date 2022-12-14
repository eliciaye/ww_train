import numpy as np
import pandas as pd
# depends on config.txt file
# experiment_name -> runs for different seeds
def get_avg5_accs(filename,ids):
    test,train=[],[]
    for i in ids:
        testi,traini=[],[]
        output_file="{}_{}.out".format(filename,i)
        with open(output_file) as f:
            print(output_file)
            for line in f:
                if "Train Acc" in line and "Final" not in line:
                    train_acc = float(line.split(' ')[-1][:-2]) # test accuracy at the end of the summary line
                    traini.append(train_acc)
                if ("Test acc" in line or "Test Acc" in line) and "Epoch 200" not in line:
                    if "Test Acc" in line:
                        test_acc = float(line.split(' ')[-1])
                    else:
                        test_acc=float(line.split(' ')[-1][4:]) # test accuracy at the end of the summary line

                    testi.append(test_acc)
        testi.sort()
        testi.reverse()
        traini.sort()
        traini.reverse()
        test.append(sum([testi[j] for j in range(5)])/5)
        train.append(sum([traini[j] for j in range(5)])/5)

    print("\ntest=np.array({})".format(test))
    for t in test:
        print(t)
   # for i,t in enumerate(test):
   #     print(ids[i],t)
    print("\ntrain=np.array({})".format(train))
    for t in train:
        print(t)
   # for i,t in enumerate(train):
   #     print(ids[i],t)

def get_accs(filename, ids):
    test,train=[],[]
    for i in ids:
        output_file="{}_{}.out".format(filename,i)
        with open(output_file) as f:
            print(output_file)
            found_test=found_train=False
            for line in f:
                if line.startswith('/data/eliciaye/val_experiments/'):
                    print(line)
                if "Epoch 200" in line:
                    found_test=True
                    test_acc=float(line.split(' ')[-1][:-2])
                    test.append(test_acc)
                if "Final Train Acc" in line:
                    found_train=True
                    train_acc=float(line.split(' ')[-1])
                    train.append(train_acc)
            if not found_test:
                test.append(0)
            if not found_train:
                train.append(0)
    print("\ntest=np.array({})".format(test))
    for t in test:
        print(t)
   # for i,t in enumerate(test):
   #     print(ids[i],t)
    print("\ntrain=np.array({})".format(train))
    for t in train:
        print(t)
   # for i,t in enumerate(train):
   #     print(ids[i],t)
   
def get_train_accs(filename,ids):
    train=[]
    for i in ids:
        output_file="{}_{}.out".format(filename,i)
        with open(output_file) as f:
            for line in f:
                if "Train Acc" in line and "Final" not in line:
                    print(line)
                    train_acc = float(line.split(' ')[-1][:-2])
                    train.append(train_acc)
    print("train_acc=",train)

def get_test_accs(filename,ids):
    for i in ids:
        test=[]
        output_file="{}_{}.out".format(filename,i)
        with open(output_file) as f:
            for line in f:
                if ("Test acc" in line or "Test Acc" in line) and "Epoch 200" not in line:
                    if "Test Acc" in line:
                        test_acc = float(line.split(' ')[-1][:-2])
                    else:
                        test_acc=float(line.split(' ')[-1][4:])
                    test.append(test_acc)
        print("test_acc=",test)

def get_train_loss(filename,ids):
    for i in ids:
        train=[]
        output_file="{}_{}.out".format(filename,i)
        with open(output_file) as f:
            for line in f:
                if "Train loss" in line:
                    train_loss= float(line.split(' ')[-1])
                    train.append(train_loss)
        print("train_loss=",train)

def get_test_loss(filename,ids):
    test=[]
    for i in ids:
        output_file="{}_{}.out".format(filename,i)
        with open(output_file) as f:
            for line in f:
                if "Test loss" in line:
                    test_acc=float(line.split(' ')[-1][5:-2])
                    test.append(test_acc)
    print("test_loss=",test)


FILENAME='resnet_avg_wdonly_58778'
# IDS=[26,31,36,41,46,27,32,37,42,47]+list(range(73,97))
# IDS=[1,6,11,16,21,2,7,12,17,22,26,31,36,41,46,27,32,37,42,47]
# IDS=[1,6,11,16,21,2,7,12,17,22,36,41,46,37,42,47]
# IDS=[3,4,5,8,9,10,13,14,15,18,19,20,23,24,25,28,29,30,33,34,35,38,39,40,43,44,45,48,49,50]
# IDS=[11,16,21,12,17,22,36,41,46,37,42,47]
IDS=list(range(73,91))
# FILENAME='resnet_wdorlr_57908'
# IDS=list(range(1,13))+list(range(37,49))
# IDS=[26,31,36,41,46,27,32,37,42,47]

get_accs(FILENAME,IDS)
# FILENAME='output'
# IDS=[58066,58067]
# get_test_accs(FILENAME,IDS)
# get_train_loss(FILENAME,IDS)