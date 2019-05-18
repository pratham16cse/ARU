from __future__ import print_function

import csv
import pandas as pd
import re
import os
import numpy as np
import math
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict

def createPickleFromRawData(params, writeToFile=False):
    df = pd.read_csv('Walmart/train.csv')
    dfStores = pd.read_csv('Walmart/stores.csv')
    dfFeatures = pd.read_csv('Walmart/features.csv')
    df = pd.merge(df, dfStores, on='Store', how='left')
    df = pd.merge(df, dfFeatures, on=['Store', 'Date'], how='left')
    df['IsHoliday'] = df['IsHoliday_x'] + df['IsHoliday_y']
    df = df.drop(['IsHoliday_x', 'IsHoliday_y'], axis=1)
    df['Date'] = pd.DatetimeIndex(df['Date'])
    numerical_features = ['Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment']
    df[numerical_features] /= df[numerical_features].max()
    df = df.fillna(0)
    df['Type'] = df['Type'].replace(['A','B','C'], [0,1,2])
    df = df.set_index(['Store', 'Dept', 'Date'])
    # Explicitly marking Holidays
    df.loc[(slice(None), slice(None), ('2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08')),'IsHoliday'] = True
    df.loc[(slice(None), slice(None), ('2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06')),'IsHoliday'] = True
    df.loc[(slice(None), slice(None), ('2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29')),'IsHoliday'] = True
    df.loc[(slice(None), slice(None), ('2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27')),'IsHoliday'] = True
    df['IsHoliday'] *= 1
    # print((df['Weekly_Sales'] < 0).sum())
    df = df.loc[df['Weekly_Sales']>=0]


    # Create individual timeseries from dataframe
    XX = list()
    lagXX = list()
    YY = list()
    for tsId, (store_dept, ts) in enumerate(df.groupby(level=['Store', 'Dept'])):
        #print(store_dept)
        ts = ts.reset_index(level=['Store', 'Dept'], drop=True) # Date is the new index now.
        ts['Day'] = ts.index.day - 1
        ts['Month'] = ts.index.month - 1
        ts['Year'] = ts.index.year - 2010
        ts['WeekOfYear'] = ts.index.weekofyear - 1
        if params.isTsId:
            ts['tsId'] = tsId
        
        Y = ts[['Weekly_Sales']].values.tolist()
        YY.append(Y)

        lags = [{'weeks':1}]
        if params.isLagFeat:
            lags += [{'weeks':52}]
        lagX = list()
        for lag in lags:
            lagX.append(ts['Weekly_Sales'].get(ts.index - pd.DateOffset(**lag)).fillna(0).reset_index(drop=True))
        lagX = pd.concat(lagX, axis=1, ignore_index=True)
        lagX = lagX.values.tolist()
        lagXX.append(lagX)

        ts = ts.drop(columns='Weekly_Sales')
        if params.noFeat:
            X = [[] for x in ts.values.tolist()]
        else:
            X = ts.values.tolist()
        XX.append(X)
        features = ts.columns.values

    if params.noFeat:
        feature_dict, emb_dict = {}, {}
    else:
        feature_dict = OrderedDict([(x,i) for i,x in enumerate(features)])
        #for k,v in feature_dict.items():
        #    print('feature_dict', k,v)
        emb_list = [('Type', (3, 1)), ('Day', (31, 10)), ('Month', (12, 6)), ('Year', (3, 1)), ('WeekOfYear', (52, 10))]
        if params.isTsId:
            emb_list += [('tsId', (len(XX), 50))]
        emb_dict = OrderedDict(emb_list)

    if writeToFile:
        with open('datasets/walmart.pkl','wb') as f:
            pickle.dump(XX, f)
            pickle.dump(lagXX, f)
            pickle.dump(YY, f)
            pickle.dump(feature_dict, f)
            pickle.dump(emb_dict, f)

    return XX, lagXX, YY, feature_dict, emb_dict


def createPickleFromRawData_old():
        df = pd.read_csv('Walmart/train.csv')
        dfStores = pd.read_csv('Walmart/stores.csv')
        dfFeatures = pd.read_csv('Walmart/features.csv')
        df = pd.merge(df, dfStores, on='Store', how='left')
        df = pd.merge(df, dfFeatures, on=['Store', 'Date'], how='left')
        df['IsHoliday'] = df['IsHoliday_x'] + df['IsHoliday_y']
        df = df.drop(['IsHoliday_x', 'IsHoliday_y'], axis=1)
        # df = pd.merge(df, dfFeatures, on=['Store','Date'], how='left')
        # print(df.loc[(df['Date'] == '2010-02-12')].shape)
        #print(df['IsHoliday'].sum().sum())
        df['Day'] = pd.DatetimeIndex(df['Date']).day
        df['Month'] = pd.DatetimeIndex(df['Date']).month
        df['Year'] = pd.DatetimeIndex(df['Date']).year

        #df.columns = [u'Store', u'Dept', u'Date', u'Weekly_Sales', u'IsHoliday', u'Day',
        #   u'Month', u'Year', u'Type', u'Size', u'Store2', u'Date2',
        #   u'Temperature', u'Fuel_Price', u'MarkDown1', u'MarkDown2', u'MarkDown3',
        #   u'MarkDown4', u'MarkDown5', u'CPI', u'Unemployment', u'IsHoliday2']

        numerical_features =['Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment']
        # numerical_features =['Size']


        # xF = ['Store','Dept','Day','Month','Year','Type','IsHoliday','isHA1','isHA2','isHA3','isHB1','isHB2','isHB3','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment']
        # xF = ['Store','Dept','Day','Month','Year','Type','IsHoliday','Size']
        # xF = ['Store','Dept','Day','Month','Year','Type','IsHoliday','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment']
        xF = ['Day','Month','Year','Type','IsHoliday','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment']

        #print "@@@@@@@@@@@@@@@@@@@@@@@len = ",len(xF)
        df = df.fillna(0)

        df['IsHoliday']*=1
        df['Type'] = df['Type'].replace(['A','B','C'], [0,1,2])
        df['Year'] = df['Year'] - 2010
        df['Month'] = df['Month'] - 1
        df['Day'] = df['Day'] - 1
        df['Store'] = df['Store'] - 1
        df['Dept'] = df['Dept'] - 1


#        for k in numerical_features:
#            df[k] = df[k]/np.max(df[k].values)

        ddf = df.loc[df['Weekly_Sales'] <= 0]
        XX = list()
        YY = list()
        count = 0
        yF = ['ScaledSales']
        salesMax = max(df['Weekly_Sales'])
        df['ScaledSales'] = df['Weekly_Sales'].values

        pair2id = dict()
        for item, shop in df[['Store','Dept']].values:
            if pair2id.get((item, shop), -1) == -1:
                pair2id[(item, shop)] = 1
            else:
                pair2id[(item, shop)] += 1

#        for cnt, ((store, dept), sales) in enumerate(pair2id.items()):
        for store in range(45):
            for dept in range(100):
                if len(ddf.loc[(ddf['Store'] == store) & (ddf['Dept'] == dept)]) == 0:
                    W = df.loc[(df['Store'] == store) & (df['Dept'] == dept)]
                    numWeeks = W.shape[0]
                    #print(W.isna().values.sum())
                    #print(store, dept, sales, sum([s<0 for s in W['ScaledSales']]))
                    #if sum([s<0 for s in W['ScaledSales']]) == 0: # No negative sales
                    print(store, dept)
                    # W['isHA1'] = W['IsHoliday'].shift(+1)
                    # W['isHA2'] = W['IsHoliday'].shift(+2)
                    # W['isHA3'] = W['IsHoliday'].shift(+3)
                    # W['isHB1'] = W['IsHoliday'].shift(-1)
                    # W['isHB2'] = W['IsHoliday'].shift(-2)
                    # W['isHB3'] = W['IsHoliday'].shift(-3)
                    W = W.fillna(0)
                    X = W[yF+xF]
                    X['ScaledSales'] = X['ScaledSales'].shift(1).fillna(0) # adding prev y features.
                    X.rename(columns={'ScaledSales':'PrevScaledSales'}, inplace=True)
                    X = X.values.tolist()
                    Y = W[yF].values.tolist()
                    XX.append(X)
                    YY.append(Y)


#        for i in range(45):
#            for j in range(100):
#                if len(ddf.loc[(ddf['Store'] == i) & (ddf['Dept'] == j)]) == 0:
#                    XXF = df.loc[(df['Store'] == i) & (df['Dept'] == j)]
#                    if testFraction:
#                        test_length = int(testFraction*(len(XXF)-1))
#                    else:
#                        test_length = int(decoder_length)
#                    train_length = len(XXF) - 1 - test_length
#                    if train_length > sequence_length and test_length >= decoder_length:
#                        XXF = df.loc[(df['Store'] == i) & (df['Dept'] == j)]
#                        # XXF['isHA1'] = XXF['IsHoliday'].shift(+1)
#                        # XXF['isHA2'] = XXF['IsHoliday'].shift(+2)
#                        # XXF['isHA3'] = XXF['IsHoliday'].shift(+3)
#                        # XXF['isHB1'] = XXF['IsHoliday'].shift(-1)
#                        # XXF['isHB2'] = XXF['IsHoliday'].shift(-2)
#                        # XXF['isHB3'] = XXF['IsHoliday'].shift(-3)
#                        XXF = XXF.fillna(0)
#                        X = XXF[xF+yF]
#                        X['ScaledSales'] = X['ScaledSales'].shift(1).fillna(0) # adding prev y features.
#                        X.rename(columns={'ScaledSales':'PrevScaledSales'}, inplace=True)
#                        X = X.values.tolist()
#                        Y = XXF[yF].values.tolist()
#                        XX.append(X)
#                        YY.append(Y)
#                        count += 1

        with open('datasets/walmart.pkl','wb') as f:
            pickle.dump(XX, f)
            pickle.dump(YY, f)


def getValuesWalmart(testFraction,sequence_length,decoder_length,modelToRun):
    with open('datasets/walmart.pkl','r') as f:
        pickle.load(XX, f)
        pickle.load(YY, f)
    XTrain, YTrain, XTest, YTest, maxYY = list(), list(), list(), list(), list()

    for i in range(len(XTrain)): # Normalizing each feature by max value along the sequence

        #XTrain[i] = np.exp(XTrain[i]).tolist()
        maxX = np.array(XTrain[i]).max(axis=0)
        XTrain[i] = (np.array(XTrain[i]) / maxX)
        XTrain[i][np.isnan(XTrain[i])] = 0
        XTrain[i][np.isinf(XTrain[i])] = 0
        XTrain[i] = XTrain[i].tolist()

        #XTest[i] = np.exp(XTest[i]).tolist()
        XTest[i] = (np.array(XTest[i]) / maxX)
        XTest[i][np.isnan(XTest[i])] = 0
        XTest[i][np.isinf(XTest[i])] = 0
        XTest[i] = XTest[i].tolist()

        #YTrain[i] = np.exp(YTrain[i]).tolist()
        maxy = np.array(YTrain[i]).max()
        maxY.append(maxy)
        YTrain[i] = (np.array(YTrain[i]) / maxy)
        YTrain[i][np.isnan(YTrain[i])] = 0
        YTrain[i][np.isinf(YTrain[i])] = 0
        YTrain[i] = YTrain[i].tolist()

        #YTest[i] = np.array(YTest[i]).tolist()
        YTest[i] = (np.array(YTest[i]) / maxy)
        YTest[i][np.isnan(YTest[i])] = 0
        YTest[i][np.isinf(YTest[i])] = 0
        YTest[i] = YTest[i].tolist()

    assert len(XTrain) == len(YTrain)
#    print('Number of timeseries:', len(XTrain))
#    print('Len of each TS:', [len(i) for i in XTrain])
#    print('XTrain[0].shape:',np.array(XTrain[0]).shape)
#    print('YTrain[0].shape:',np.array(YTrain[0]).shape)
#    print('XTest[0].shape',np.array(XTest[0]).shape)
#    print('YTest[0].shape',np.array(YTest[0]).shape)

    return XTrain, YTrain, XTest, YTest, count, maxY, len(XTrain[0][0])


def getWalmartEmbeddingTransform(input1,isEmbedding):
        store_embed_size = 5 #45
        dept_embed_size = 10 #99
        day_embed_size = 10 #31
        month_embed_size = 6 #12
        year_embed_size = 1 #3
        type_embed_size = 1 #3
        hol_embed_size = 1 #2

        structure = []

        #prev Y
        prevY = tf.expand_dims(tf.cast(input1[:,:,0],tf.float32), axis=2)
        structure.append(prevY)

#        #store
#        v_store_index = tf.Variable(tf.random_uniform([45,store_embed_size], -1.0, 1.0))
#        em_store_index = tf.nn.embedding_lookup(v_store_index, tf.cast(input1[:,:,1],tf.int32))
#        if isEmbedding:
#                structure.append(em_store_index)
#
#        #dept
#        v_dept_index = tf.Variable(tf.random_uniform([99,dept_embed_size], -1.0, 1.0))
#        em_dept_index = tf.nn.embedding_lookup(v_dept_index, tf.cast(input1[:,:,2],tf.int32))
#        if isEmbedding:
#                structure.append(em_dept_index)

        #day
        v_day_index = tf.Variable(tf.random_uniform([31,day_embed_size], -1.0, 1.0, seed=12))
        em_day_index = tf.nn.embedding_lookup(v_day_index, tf.cast(input1[:,:,1],tf.int32))
        structure.append(em_day_index)

        #month
        v_month_index = tf.Variable(tf.random_uniform([12,month_embed_size], -1.0, 1.0, seed=12))
        em_month_index = tf.nn.embedding_lookup(v_month_index, tf.cast(input1[:,:,2],tf.int32))
        structure.append(em_month_index)

        #year
        v_year_index = tf.Variable(tf.random_uniform([3,year_embed_size], -1.0, 1.0, seed=12))
        #v_year_index = tf.Print(v_year_index, [ v_year_index, input1[:,:,3]], message="HERE", summarize=30000)
        em_year_index = tf.nn.embedding_lookup(v_year_index, tf.cast(input1[:,:,3],tf.int32))
        structure.append(em_year_index)

        #type
        v_type_index = tf.Variable(tf.random_uniform([3,type_embed_size], -1.0, 1.0, seed=12))
        em_type_index = tf.nn.embedding_lookup(v_type_index, tf.cast(input1[:,:,4],tf.int32))
        if isEmbedding:
            structure.append(em_type_index)

        #hol
        v_hol_index = tf.Variable(tf.random_uniform([2,hol_embed_size], -1.0, 1.0, seed=12))
        em_hol_index = tf.nn.embedding_lookup(v_hol_index, tf.cast(input1[:,:,5],tf.int32))
        structure.append(em_hol_index)

        #hol1
        # v_hol_index = tf.Variable(tf.random_uniform([2,hol_embed_size], -1.0, 1.0))
        # em_hol_index = tf.nn.embedding_lookup(v_hol_index, tf.cast(input1[:,:,7],tf.int32))
        # structure.append(em_hol_index)

        # #hol2
        # v_hol_index = tf.Variable(tf.random_uniform([2,hol_embed_size], -1.0, 1.0))
        # em_hol_index = tf.nn.embedding_lookup(v_hol_index, tf.cast(input1[:,:,8],tf.int32))
        # structure.append(em_hol_index)

        # #hol3
        # v_hol_index = tf.Variable(tf.random_uniform([2,hol_embed_size], -1.0, 1.0))
        # em_hol_index = tf.nn.embedding_lookup(v_hol_index, tf.cast(input1[:,:,9],tf.int32))
        # structure.append(em_hol_index)

        # #hol4
        # v_hol_index = tf.Variable(tf.random_uniform([2,hol_embed_size], -1.0, 1.0))
        # em_hol_index = tf.nn.embedding_lookup(v_hol_index, tf.cast(input1[:,:,10],tf.int32))
        # structure.append(em_hol_index)

        # #hol5
        # v_hol_index = tf.Variable(tf.random_uniform([2,hol_embed_size], -1.0, 1.0))
        # em_hol_index = tf.nn.embedding_lookup(v_hol_index, tf.cast(input1[:,:,11],tf.int32))
        # structure.append(em_hol_index)

        # #hol6
        # v_hol_index = tf.Variable(tf.random_uniform([2,hol_embed_size], -1.0, 1.0))
        # em_hol_index = tf.nn.embedding_lookup(v_hol_index, tf.cast(input1[:,:,12],tf.int32))
        # structure.append(em_hol_index)

        #num_f_index
        num_f_index = tf.cast(input1[:,:,6:],tf.float32)
        structure.append(num_f_index)

        input11 = tf.concat(structure, axis=2)
        #print "@@@@@@@@@@@@@@@@@@@@@@@@transform",input11
        return input11

if __name__ == '__main__':
    testFraction = 0
    sequence_length = 24
    decoder_length = 16
    modelToRun = 'baseline'
    createPickleFromRawData(params, writeToFile=True)
    #XTrain, YTrain, XTest, YTest, count, maxY, numFW = getValuesWalmart(testFraction,sequence_length,decoder_length,modelToRun)
    tsId = np.random.randint(len(XTrain))
    X_tr = np.array(XTrain[tsId])
    y_tr = np.array(YTrain[tsId])[:,0].tolist()
    X_test = np.array(XTest[tsId])
    y_test = np.array(YTest[tsId])[:,0].tolist()

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
