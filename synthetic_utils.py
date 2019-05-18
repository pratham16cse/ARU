from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import pickle
from collections import OrderedDict
import pandas as pd
np.random.seed(12)

numTs = 10
seq_len = 1000
#numF = 4
numF = 2
hW_size = 5
isLinear = True

def createPickleFromRawData(params, writeToFile=False):
    numF = 2
    if params.isTsId:
        numF = numF + 1
    seq_len = params.synth_seq_len
    if isLinear:
        w = np.random.uniform(-1.0, 1.0, size=[params.synth_num_ts, numF])
    else:
        w = np.random.uniform(-3.0, 3.0, size=[params.synth_num_ts, hW_size])
    #w = np.concatenate((w, np.zeros((params.synth_num_ts, 1))), axis=1)
    b = np.random.uniform(-1.0, 1.0, size=[params.synth_num_ts])
    yprev = np.random.uniform(-1.0, 1.0, size=[params.synth_num_ts, 1])
    #X = np.random.uniform(-1.0, 1.0, size=[params.synth_num_ts, seq_len, numF])
    date_range = pd.date_range('2018-06-01','2018-06-30', freq='H')
    indices = np.random.randint(low=0, high=len(date_range), size=(params.synth_num_ts))
    X = np.zeros((params.synth_num_ts, seq_len, numF))
    for i, ind in enumerate(indices):
        sts = date_range[ind]
        x = pd.DataFrame(index=pd.date_range(sts, sts+seq_len-1, freq='H'))
        x['hod'] = x.index.hour
        x['dow'] = x.index.weekday
        X[i, :, :2] = x.values
    #print(X[0,:50,:])
    if params.isTsId:
        X[:, :, 2] = np.tile(np.expand_dims(np.arange(X.shape[0]), axis=1), [1, X.shape[1]])

    df = np.zeros((params.synth_num_ts, seq_len, 1))
    hW = np.random.uniform(-4.0, 4.0, size=[numF, hW_size])
    for t in range(seq_len):
        xnew = X[:, t]
        if isLinear == False:
            y_latent = xnew.dot(hW)
            hW2 = np.random.uniform(-4.0, 4.0, size=[hW_size, 1])
            ynew = 1 / (1 + np.exp(-y_latent.dot(hW2)))
            ynew = np.squeeze(ynew, axis=1)
        else:
            ynew = np.sum(w*xnew, axis=1) + b + np.random.normal(scale=0.05, size=[params.synth_num_ts])
        df[:, t, 0] = ynew
    df = np.concatenate((df, X), axis=2)
    #print(df.shape)

    Y = df[:, 1:, 0]
    XX = df[:, 1:, 1:]
    print(XX.shape, Y.shape)
    # Normalize XX
#    for i in range(len(XX)):
#        maxX = np.max(XX[i, :, :], axis=0)
#        XX[i, :, :] /= maxX
    num_lags = []
    if params.isLagFeat:
        num_lags += [17, 18]
    lagXX = np.expand_dims(df[:, :-1, 0], axis=2)
    print(Y[:, :5])
    print(lagXX[:, :5, 0])
    print(XX[8, :10, :])#, np.isnan(XX).any())
    for i, lag in enumerate(num_lags):
        # lagXX[:, :, i] = np.concatenate([np.zeros((Y.shape[0], 16+num_lags)), Y[:, :-16-num_lags]], axis=1)
        lagXX[:, :, i] = np.pad(Y, ((0, 0), (lag, 0)), mode='constant')[:, :-lag]
        # print(lagXX[0])
    YY = np.expand_dims(Y, axis=2)
    
    feature_dict = OrderedDict([(str(i), i) for i in range(numF)])
    emb_dict = OrderedDict([('0', (24, 3)), ('1', (7, 2))])

    with open('datasets/synthetic.pkl','wb') as f:
        pickle.dump(XX, f)
        pickle.dump(lagXX, f)
        pickle.dump(YY, f)
        pickle.dump(feature_dict, f)
        pickle.dump(emb_dict, f)

    return XX, lagXX, YY, feature_dict, emb_dict
        

#    for i in range(len(XTrain)):
#
#        #XTrain[i] = np.exp(XTrain[i]).tolist()
#        maxX = np.array(XTrain[i]).max(axis=0)
#        XTrain[i] = (np.array(XTrain[i]) / maxX)
#        XTrain[i][np.isnan(XTrain[i])] = 0
#        XTrain[i][np.isinf(XTrain[i])] = 0
#        XTrain[i] = XTrain[i].tolist()
#
#        #XTest[i] = np.exp(XTest[i]).tolist()
#        XTest[i] = (np.array(XTest[i]) / maxX)
#        XTest[i][np.isnan(XTest[i])] = 0
#        XTest[i][np.isinf(XTest[i])] = 0
#        XTest[i] = XTest[i].tolist()
#
#        #YTrain[i] = np.exp(YTrain[i]).tolist()
#        maxY = np.array(YTrain[i]).max()
#        maxYY.append(maxY)
#        YTrain[i] = (np.array(YTrain[i]) / maxY)
#        YTrain[i][np.isnan(YTrain[i])] = 0
#        YTrain[i][np.isinf(YTrain[i])] = 0
#        YTrain[i] = YTrain[i].tolist()
#
#        #YTest[i] = np.array(YTest[i]).tolist()
#        YTest[i] = (np.array(YTest[i]) / maxY)
#        YTest[i][np.isnan(YTest[i])] = 0
#        YTest[i][np.isinf(YTest[i])] = 0
#        YTest[i] = YTest[i].tolist()

    #print(np.array(XTrain).shape, np.array(YTrain).shape, np.array(XTest).shape, np.array(YTest).shape)

    #return XTrain, YTrain, XTest, YTest, count, maxYY, len(XTrain[0][0])


def plotData():
        data = pd.read_csv('WeeklySales/data_weekly_sales.csv')
        feat = list()
        for i in range(52):
            feat.append('W'+str(i))
        data = data[feat]
        plotIndex = np.random.randint(0,data.shape[0])
        plt.plot(range(1,len(feat)+1), data.loc[plotIndex], 'bs')
        plt.plot(range(1,len(feat)+1), data.loc[plotIndex], 'b-', linewidth=2)
        plt.show()


if __name__ == '__main__':
    testFraction = 0.2
    modelToRun = 'baseline'
    createPickleFromRawData(params, writeToFile=True)
    #XTrain1,YTrain1,PrevYTrain1,XTest1,YTest1,PrevYTest1,count,maxYY,numFW = getValuesW(testFraction,modelToRun)
    #XTrain2,YTrain2,XTest2,YTest2,count2,maxY2,numFW2 = prepareSyntheticData(testFraction, modelToRun)
    #plotData()
