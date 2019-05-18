import csv
import pandas as pd
import re
import numpy as np
import math
import pickle
from collections import OrderedDict

def createPickleFromRawData(params, writeToFile=False):
    df = pd.read_csv('Carparts/carparts.csv')
    df = df.dropna()
    df_1 = df.loc[(df > 0).sum(axis=1)>10,:]
    df_2 = df_1.loc[(df_1.iloc[:,:16]>=1).sum(axis=1)>0]
    df_3 = df_2.loc[(df_2.iloc[:,37:]>=1).sum(axis=1)>0]
    df = df_3

    timestamps = pd.date_range(start='1998-01-31', end='2002-03-31', freq='M')
    df = df.T
    df.columns = df.iloc[0,:]
    df = df.drop(index='Part')
    df.index = timestamps

    lags = [{'months':1}]
    if params.isLagFeat:
        lags += [{'months':12}]
    lagXX = list()
    for lag in lags:
        shift_offset = lag['months']
        lagXX.append(df.shift(shift_offset).fillna(0))

    lagXX = np.transpose(np.stack(lagXX, axis=2), [1, 0, 2]).tolist()

    XX = df.index.month * 1.0 - 1
    XX = np.tile(np.expand_dims(XX.values, axis=1), [1, df.shape[1]]).T
    XX = np.expand_dims(XX, axis=2)
    if params.isTsId:
        tsIds = np.tile(np.expand_dims(np.arange(len(XX)), axis=1), [1, XX.shape[1]])
        tsIds = np.expand_dims(tsIds, axis=2)
        XX = np.concatenate([XX, tsIds], axis=2)
    XX = XX.tolist()
    YY = df.values.T
    YY = np.expand_dims(YY, axis=2).tolist()


    feature_dict = OrderedDict([('monthOfYear', 0)])
    emb_dict = OrderedDict([('monthOfYear', (12, 6))])
    if params.isTsId:
        feature_dict['tsId'] = 1
        emb_dict['tsId'] = (len(YY), 50)

    with open('datasets/carparts.pkl','wb') as f:
        pickle.dump(XX, f)
        pickle.dump(lagXX, f)
        pickle.dump(YY, f)
        pickle.dump(feature_dict, f)
        pickle.dump(emb_dict, f)

    return XX, lagXX, YY, feature_dict, emb_dict


def getValues(encoder_length,decoder_length,test_length):
	sequence_length = encoder_length + decoder_length
	df = pd.read_csv('Carparts/carparts.csv')
	df = df.values
	count = 0

	X = []
	for i in range(df.shape[0]):
	    l = df[i][~np.isnan(df[i])]
	    if l.shape[0] == 52:
	        count1 = 0
	        for k in l[1:]:
	            if k > 0:
	                count1 = count1+1
	        if count1 >= 10:
	            flag = False
	            c1 = 0
	            for k in l[1:16]:
	                if k > 0:
	                    c1 = c1+1
	                    if c1 >= 1:
	                        flag = True
	                        break
	            c1 = 0
	            if flag == True:
	                flag = False
	                for k in l[37:]:
	                    if k > 0:
	                        c1 = c1+1
	                        if c1 >= 1:
	                            flag = True
	                            break
	            if flag == True:
	                count = count+1
	                l_ = []
	                for i in l:
	                    l_.append([i])
	                X.append(l_)
	df = np.array(X)

	XXTrain = []
	YYTrain = []
	PrevYYTrain = []
	XXTest = []
	YYTest = []
	PrevYYTest = []

	count = 0
	graph = {}

	for i in range(df.shape[0]):
	    l = df[i][~np.isnan(df[i])]

	    s = 1
	    if graph.has_key(l[0]) == False:
	        graph[l[0]] = count
	        count = count+1
	    
	    curY = l[2:-test_length]
	    prevY = l[1:-test_length-1]
	    X = []
	    Y = []
	    PY = []
	    for k in range(len(curY)):
	        X.append([graph[l[0]],k])
	        Y.append([curY[k]])
	        PY.append([prevY[k]])
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
	        X.append([graph[l[0]],k+ll])
	        Y.append([curY[k]])
	        PY.append([prevY[k]])
	    XXTest.append(X)
	    YYTest.append(Y)
	    PrevYYTest.append(PY)
	print(np.shape(XXTest[0])[1])
	return XXTrain,YYTrain,PrevYYTrain,XXTest,YYTest,PrevYYTest,count,np.shape(XXTest[0])[1]

def main():
    XX, lagXX, YY, feature_dict, emb_dict = createPickleFromRawData(params, writeToFile=True)

if __name__ == '__main__':
    main()
