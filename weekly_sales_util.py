import csv
import pandas as pd
import re
import os
import numpy as np
import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle


def getValuesW(testFraction, model_to_run):
        df = pd.read_csv('WeeklySales/data_weekly_sales.csv')
        feat = ['Product_Code']
        for i in range(52):
            feat.append('W'+str(i))

        XX = []
        for kk in df[feat].values:
            XX.append(kk[1:])

        XXTrain = []
        YYTrain = []
        PrevYYTrain = []
        XXTest = []
        YYTest = []
        PrevYYTest = []
        maxYY = []

        for i in range(len(XX)):
            l = XX[i]
            test_length = int(testFraction*(len(l)-1))
            curY = l[2:-test_length]
            prevY = l[1:-test_length-1]
            X = []
            Y = []
            PY = []
            maxY = 1
            for k in range(len(curY)):
                X.append([i,k])
                Y.append([curY[k]])
                PY.append([prevY[k]])
                if maxY < curY[k]:
                    maxY = curY[k]
                if maxY < prevY[k]:
                    maxY = prevY[k]
            
            maxYY.append(maxY)

#            Y = [[x[0]] for x in Y]
#            PY = [[x[0]] for x in PY]
            if model_to_run == "aru":
                Y = [[x[0]] for x in Y]
                PY = [[x[0]] for x in PY]
            else:
                Y = [[x[0] / float(maxY)] for x in Y]
                PY = [[x[0] / float(maxY)] for x in PY]

            XXTrain.append(X)
            YYTrain.append(Y)
            PrevYYTrain.append(PY)

            ll = len(curY)
            curY = l[-test_length:]
            prevY = l[-test_length-1:]
            X = []
            Y = []
            PY = []
            for k in range(len(curY)):
                X.append([i,k+ll])
                Y.append([curY[k]])
                PY.append([prevY[k]])

#            Y = [[x[0]] for x in Y]
#            PY = [[x[0]] for x in PY]
            if model_to_run == "aru":
                Y = [[x[0]] for x in Y]
                PY = [[x[0]] for x in PY]
            else:
                Y = [[x[0] / float(maxY)] for x in Y]
                PY = [[x[0] / float(maxY)] for x in PY]

            XXTest.append(X)
            YYTest.append(Y)
            PrevYYTest.append(PY)

        YY = []
        for i in range(np.shape(YYTrain)[0]):
            Y = [y[0] for y in YYTrain[i]]
            YT = [y[0] for y in YYTest[i]]
            Y = Y+YT
            YY.append(Y)

        countF = 0
        YYTrainF = []
        XXTrainF = []
        PrevYYTrainF = []
        XXTestF = []
        YYTestF = []
        PrevYYTestF = []
        maxYYF = []
        for i in range(len(YY)):
            y = YY[i]
            count1 = 0
            for k in y:
                if k > 0:
                    count1 = count1+1
            if count1 >= 10:
                flag = False
                c1 = 0
                for k in y[0:16]:
                    if k > 0:
                        flag = True
                        break
                c1 = 0
                if flag == True:
                    flag = False
                    for k in y[37:]:
                        if k > 0:
                            flag = True
                            break
                if flag == True:
                    YYTrainF.append(YYTrain[i])
                    XXTrainF.append(XXTrain[i])
                    PrevYYTrainF.append(PrevYYTrain[i])
                    XXTestF.append(XXTest[i])
                    YYTestF.append(YYTest[i])
                    PrevYYTestF.append(PrevYYTest[i])
                    maxYYF.append(maxYY[i])
                    countF+=1
                    
        XXTrain,YYTrain,PrevYYTrain,XXTest,YYTest,PrevYYTest,count,maxYY = XXTrainF,YYTrainF,PrevYYTrainF,XXTestF,YYTestF,PrevYYTestF,countF,maxYYF

        print('Number of timeseries:', len(XXTrain))
        print('XXTrain[0].shape:',np.array(XXTrain[0]).shape)
        print('YYTrain[0].shape:',np.array(YYTrain[0]).shape)
        print('PrevYYTrain[0].shape',np.array(PrevYYTrain[0]).shape)
        print('XXTest[0].shape',np.array(XXTest[0]).shape)
        print('YYTest[0].shape',np.array(YYTest[0]).shape)
        print('PrevYYTest[0].shape',np.array(PrevYYTest[0]).shape)

        return XXTrain,YYTrain,PrevYYTrain,XXTest,YYTest,PrevYYTest,count,maxYY,np.array(XXTest[0]).shape[1]


def createPickleFromRawData():
    df = pd.read_csv('WeeklySales/data_weekly_sales.csv')
    feat = ['W'+str(i) for i in range(52)]
    df = df[feat]*1.0
    df = df.loc[(df != 0).sum(axis=1)>10] # Select all timeseries which contain at least 10 nonzero entries.
    df = df.loc[(df.iloc[:,:16]>0).sum(axis=1)>0] # Timeseries whose first 16 values contain at least one nonzero entry.
    df = df.loc[(df.iloc[:,-16:]>0).sum(axis=1)>0] # Timeseries whose last 16 values contain at least one nonzero entry.
    Y = df
    # lagX = df.iloc[:,:-1]
    lagX = pd.concat([pd.DataFrame(0, index=df.index, columns=['W-1']), df.iloc[:,:-1]], axis=1)
    lagXX = np.expand_dims(lagX.values, axis=2).tolist()

    YY = Y.values
    YY = np.expand_dims(YY, axis=2).tolist()
    XX = list()
    for Y in YY:
        X = list()
        for y in Y:
            X.append([])
        XX.append(X)
    #XX = [[[]] for y in Y for Y in YY]
    #print(XX)
    #print(YY)

    feature_dict = {}
    emb_dict = {}

    with open('datasets/weeklySales.pkl','wb') as f:
        pickle.dump(XX, f)
        pickle.dump(lagXX, f)
        pickle.dump(YY, f)
        pickle.dump(feature_dict, f)
        pickle.dump(emb_dict, f)

    return XX, lagXX, YY, feature_dict, emb_dict


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
    decoder_length = 16
    modelToRun = 'baseline'
    createPickleFromRawData()
    #XXTrain1,YYTrain1,PrevYYTrain1,XXTest1,YYTest1,PrevYYTest1,count,maxYY,numFW = getValuesW(testFraction,modelToRun)
    #XXTrain, YYTrain, XXTest, YYTest, count, maxY, numFW = prepareWeeklySales(testFraction, decoder_length, modelToRun)
    #plotData()
    tsId = np.random.randint(len(XXTrain))
    X_tr = np.array(XXTrain[tsId])
    y_tr = np.array(YYTrain[tsId])[:,0].tolist()
    X_test = np.array(XXTest[tsId])
    y_test = np.array(YYTest[tsId])[:,0].tolist()

    #---- Plotter Block Start -----#
    plt.plot([len(y_tr)-1, len(y_tr)],[y_tr[-1], y_test[0]], 'r') # Connecting conditioning and prediction range
    #plt.plot((len(y_tr)-1+len(y_tr))*1.0/2, 'r') # TODO:Separator line between conditioning and prediction range
    plt.plot(range(len(y_tr)), y_tr, 'b') # Plotting conditioning range
    #plt.plot(range(len(y_tr)), y_tr, 'bo') # plotting conditioning range (with marker)
    plt.plot(range(len(y_tr), len(y_tr)+len(y_test)), y_test, 'r') # Plotting prediction range
    #plt.plot(range(len(y_tr), len(y_tr)+len(y_test)), y_test, 'ro') # Plotting prediction range (with marker)
    plt.show()
    #---- Plotter Block End -----#
#    for i in range(X.shape[1]):
#        #pltX = X[:,i].tolist()
#        #print(pltX)
#        #print(y)
#        plt.plot(y, 'b')
#        plt.plot(y, 'k*')
#        plt.show()
