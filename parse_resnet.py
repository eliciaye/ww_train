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
    test=[]
    for i in ids:
        output_file="{}_{}.out".format(filename,i)
        with open(output_file) as f:
            for line in f:
                if ("Test acc" in line or "Test Acc" in line) and "Epoch 200" not in line:
                    if "Test Acc" in line:
                        test_acc = float(line.split(' ')[-1])
                    else:
                        test_acc=float(line.split(' ')[-1][4:])
                    test.append(test_acc)
    print("test_acc=",test)

def get_train_loss(filename,ids):
    train=[]
    for i in ids:
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
FILENAME='resnet_evals_55909'
IDS=[1,6,11,16,21,2,7,12,17,22]
# IDS=list(range(79,97))
# IDS=[13,14,15,18,23,24,25]
get_accs(FILENAME,IDS)
