'''
Description:
    Training a ATCNN model or applying a fitted ATCNN model to prediction
Usage:
    Training:
      python train.py -t  --datapath=the path of your dataset

    Test:
      python train.py -p --datapath=the path of your dataset

    For more information type 
      python train.py -h

Author:
    Qunchao Tong <tongqunchao@sina.com>

Date: 2019.08.22
'''
import numpy as np
import keras
from keras.models import load_model
from tc_II_predict import get_component_vector
import argparse, sys
import matplotlib.pyplot as plt

from model import ATCNN_Ef_model, ATCNN_Tc_model

def read_input(train_setfile, readlabel):
    f = open(train_setfile,'r')
    x_data = []
    y_data = []
    formula = []
    while True:
        line = f.readline().split()
        if len(line) == 0:
            break

        try:
            ix = line[0]
        except(IndexError):
            print('IndexError Ignore')
            continue
        try:
            x = get_component_vector(ix)
        except: 
            print('Format Error:', '\"',ix, '\"', 'Ignore')
            continue
        x_data.append(x)
        formula.append(ix)
        if (readlabel):
            try:
                iy = line[1]
            except(IndexError):
                print('IndexError Ignore')
                continue
            try:
                y = float(iy)
            except: 
                print('Format Error:', '\"', iy, '\"', 'Ignore')
                continue
            y_data.append(y)
    return x_data, y_data, formula

def data_split(x_data, y_data, split_ratio):
    if (len(x_data) != len(y_data)):
        print('The size of Training set and Test set is different!!!\nPlease check your dataset')
        sys.exit(0)
    x_train = []
    y_train = []
    x_predict = []
    y_predict = []
    for i in range(len(x_data)):
        x = np.random.rand()
        if x < split_ratio:
            x_train.append(x_data[i])
            y_train.append(y_data[i])
        else:
            x_predict.append(x_data[i])
            y_predict.append(y_data[i])
    return x_train, y_train, x_predict, y_predict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default=None, help='The path of dataset file')
    parser.add_argument('--train', '-t', action='store_true', help='Training ATCNN model?')
    parser.add_argument('--predict', '-p', action='store_true', help='Applying ATCNN to prediction?')
    parser.add_argument('--readlabel', '-r', action='store_true', help='Whether read the label?')
    parser.add_argument('--niter', type=int, default=2500, help='Number of epochs to training ATCNN')
    parser.add_argument('--ratio', type=float, default=0.8, help='The ratio of training set to data set')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch Size')
    opt = parser.parse_args()
    
    model = ATCNN_Ef_model()
    # model = ATCNN_Tc_model()
    print(model.summary())
    
    if opt.train:
        opt.readlabel = True
        x_data, y_data, _ = read_input(opt.datapath, opt.readlabel)
        x_train, y_train, x_predict, y_predict = data_split(x_data, y_data, split_ratio=opt.ratio)
        x_train = np.reshape(x_train,(len(x_train),10,10,1))
        x_predict = np.reshape(x_predict,(len(x_predict),10,10,1))
        y_train = np.array(y_train)
        y_predict = np.array(y_predict)
        model.fit(x_train, y_train, validation_split=0.02, batch_size=opt.batchsize, epochs=opt.niter)
        model.save('model.keras')
        loss = model.evaluate(x_predict, y_predict, batch_size=opt.batchsize)
        y_calc = model.predict(x_predict, batch_size=opt.batchsize)
        print('test set loss:',loss)
    
    if opt.predict:
        model = load_model('model.keras')
        x_predict, y_predict, formula = read_input(opt.datapath, opt.readlabel)
        n_sample = len(x_predict)
        if n_sample < opt.batchsize:
            opt.batchsize = n_sample
        x_predict = np.reshape(x_predict,(len(x_predict),10,10,1))     
        y_calc = model.predict(x_predict, batch_size=opt.batchsize).reshape(-1)

        if (opt.readlabel):
            y_predict = np.array(y_predict, float)

        f = open('results.dat','w')
        for i in range(n_sample):
            if (opt.readlabel):
                f.write(' %s %15.10f %15.10f\n' % (formula[i], y_calc[i], y_predict[i]))
            else:
                f.write(' %s %15.10f \n' % (formula[i], y_calc[i]))
        if (opt.readlabel):
            loss = model.evaluate(x_predict, y_predict, batch_size=opt.batchsize)
            print('The MAE:',loss)
