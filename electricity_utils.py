from __future__ import print_function

import pandas as pd
import pickle
from itertools import chain
from collections import OrderedDict
import tensorflow as tf
import numpy as np

alpha_list = [0.1, 0.25, 0.5, 0.9, 0.99]
alpha_list = [(1.0 - alpha) for alpha in alpha_list]

def createPickleFromRawData(params, writeToFile=False):
    data = pd.read_csv('electricity/LD2011_2014.txt', sep=";", index_col=0, parse_dates=True)
    data.index = data.index - pd.DateOffset(minutes=5)
    num_timeseries = data.shape[1]
    data = data.resample('H').mean()
    # xF = ['Hour', 'DayOfWeek']
    clients = data.columns[0:370].values.tolist()
    data['Hour'] = data.index.hour# - 1
    data['Day'] = data.index.day - 1
    data['Month'] = data.index.month - 1
    data['DayOfWeek'] = data.index.dayofweek
    data['Weekend'] = ((data['DayOfWeek'] == 5) | (data['DayOfWeek'] == 6)) * 1
    xF = ['Hour', 'Day', 'Month', 'DayOfWeek', 'Weekend']
    if params.isTsId: xF += ['tsId']
    XX = list()
    lagXX = list()
    YY = list()
    for i, client in enumerate(clients):
        print(i)
        ts = np.trim_zeros(data.loc[:,client], trim='f')
        ts = ts.to_frame()

        lags = [{'hours':1}]
        if params.isLagFeat:
            lags += [{'weeks':1}, {'weeks':2}, {'years':1}]
        lagX = list()
        for lag in lags:
            lagX.append(ts.iloc[:, 0].get(pd.DatetimeIndex(ts.index) - pd.DateOffset(**lag)).fillna(0).reset_index(drop=True))
        lagX = pd.concat(lagX, axis=1, ignore_index=True)
        lagX = lagX.values.tolist()
        lagXX.append(lagX)

        Y = ts.values.tolist()
        YY.append(Y)

        # ts['Hour'] = pd.DatetimeIndex(ts.index).hour
        # ts['Day'] = pd.DatetimeIndex(ts.index).day - 1
        # ts['Month'] = pd.DatetimeIndex(ts.index).month - 1
        # ts['Year'] = pd.DatetimeIndex(ts.index).year - 2011
        # ts['DayOfWeek'] = pd.DatetimeIndex(ts.index).dayofweek
        # ts['Weekend'] = ((ts['DayOfWeek'] == 5) | (ts['DayOfWeek'] == 6)) * 1
        if params.isTsId:
            ts['tsId'] = i
        X = data.loc[ts.index, xF].values.tolist()
        XX.append(X)

#        print(X[:10])
#        print('---------')
#        print(lagX[:10])
#        print('---------')
#        print(Y[:10])
#        print('------------------------')

    # Create feature_dict and emb_dict
    features = xF
    feature_dict = OrderedDict([(x, i) for i,x in enumerate(features)])
    emb_list = OrderedDict([('Hour', (24, 9)), ('Day', (31, 10)), ('month', (12, 6)), ('DayOfWeek', (7, 4))])

    if params.isTsId:
        emb_list.append(('tsId', (370, 20)))
    #('minute', (4, 2)),
    #('hour', (24, 9)),
    #('day', (31, 10)),
    #('month', (12, 6)),
    #('year', (5, 3)),
    #('dayofweek', (7, 4)),
    #('tsId', (370, 20)),
    emb_dict = OrderedDict(emb_list)

    if writeToFile:
        with open('datasets/electricity.pkl','wb') as f:
            pickle.dump(XX, f)
            pickle.dump(lagXX, f)
            pickle.dump(YY, f)
            pickle.dump(feature_dict, f)
            pickle.dump(emb_dict, f)

    return XX, lagXX, YY, feature_dict, emb_dict


def createPickleFromRawData_old(params):
    df = pd.read_csv('electricity/LD2011_2014.txt', sep=';')
    
    df['Date'] = pd.DatetimeIndex(df['Date'])
    df.rename(columns={'Date':'TimeStamp'}, inplace=True)
    df['TimeStamp'] = df['TimeStamp'] - pd.DateOffset(minutes=5) # To make sure that correct records aggregate
    df['Minute'] = pd.DatetimeIndex(df['TimeStamp']).minute
    df['Hour'] = pd.DatetimeIndex(df['TimeStamp']).hour
    df['Day'] = pd.DatetimeIndex(df['TimeStamp']).day - 1
    df['Month'] = pd.DatetimeIndex(df['TimeStamp']).month - 1
    df['Year'] = pd.DatetimeIndex(df['TimeStamp']).year - 2011
    df['DayOfWeek'] = pd.DatetimeIndex(df['TimeStamp']).dayofweek
    df['Weekend'] = ((df['DayOfWeek'] == 5) | (df['DayOfWeek'] == 6)) * 1

    clients = df.columns[1:371].values.tolist()
    xF = ['Minute', 'Hour', 'Day', 'Month', 'Year', 'DayOfWeek', 'Weekend']
    grpby_xF = ['Hour', 'Day', 'Month', 'Year']
    hr_xF = xF[1:]

    # Generate hourly data from 15-minute-ly data
    df = df[clients+xF+['TimeStamp']]
    agg_dict = dict()
    for client in clients:
        agg_dict[client] = 'sum'
    for x in hr_xF:
        agg_dict[x] = 'mean'
    agg_dict['TimeStamp'] = 'max'
    df = df.groupby(grpby_xF)[clients + hr_xF].agg(agg_dict)
    df = df.reset_index(drop=True)
    df = df.sort_values(['Year', 'Month', 'Day', 'Hour'])
    df = df.reset_index(drop=True)
    df = df[clients + hr_xF + ['TimeStamp']]
    df['TimeStamp'] = pd.DatetimeIndex(df['TimeStamp']).floor('h') # Remove minute information
    df = df.loc[df.shape[0]-26304:]

    XX = list()
    lagXX = list()
    YY = list()
    for tsId, client in enumerate(clients):
        print(client)
        cols = [client]+hr_xF+['TimeStamp'] if params.isLagFeat else [client]+hr_xF
        dfc = df[cols]
        nz_ind = dfc[client].nonzero()[0][0]
        dfc = dfc.loc[nz_ind:]
        dfc = dfc[cols]
        #if params.isLagFeat:
        dfc = dfc.reset_index(drop=True)

        if params.isTsId:
            dfc['tsId'] = tsId # TODO: Add timeseries_id through reader, too slow.
            cols = [client] + hr_xF + ['tsId']
        else:
            cols = [client] + hr_xF

        X = dfc[cols]
        # print(X.head())
        Y = dfc[[client]] # A column representing consumption for a client
        X[client] = X[client].shift(1).fillna(0) # adding prev y features
        X.rename(columns={client:client+'_Prev'}, inplace=True)

#        # Adding SRU Features -------------
#        emaF = ['EMA_{}'.format(alpha) for alpha in alpha_list]
#        for alpha in alpha_list: # SRU Features added
#            X['EMA_{}'.format(alpha)] = X[client+'_Prev'].loc[1:].ewm(alpha=alpha, adjust=False).mean()
#        X = X.fillna(0)
#        # print(X.head(10))


        if params.isLagFeat:
            # Creating lag features -------------
            date2idx = dict()
            for i, ts in enumerate(dfc['TimeStamp']):
                date2idx[ts] = i
    
            # lagF = ['hr_2', 'hr_3', 'd_1', 'd_2', 'd_3', 'w_1', 'w_2', 'w_3', 'q_1', 'q_2', 'q_3', 'y_1', 'y_2', 'y_3']
            # lags = [{'hours':2}, {'hours':3}, {'days':1}, {'days':2}, {'days':3}, {'weeks':1}, {'weeks':2}, {'weeks':3},
            #         {'months':3}, {'months':6}, {'months':9}, {'years':1}, {'years':2}]
            lags = [{'weeks':1}, {'weeks':2}, {'years':1}]
            lag_indices = np.zeros((X.shape[0], len(lags))) - 1
            for i, ts in enumerate(dfc['TimeStamp']):
                if i % 1000 == 0:
                    print(i)
                for j, offset in enumerate(lags):
                    lag_indices[i, j] = date2idx.get(ts - pd.DateOffset(**offset), -1)
            # print(lag_indices[-500:, :])
            lagX = np.zeros((X.shape[0], len(lags)))
            for i in range(len(lags)):
                if np.sum(lag_indices[:, i] > 0) > 0:
                    lagX[:, i] = np.squeeze(Y.loc[lag_indices[:, i]], axis=1)
                lagX[lag_indices[:, i]<0, i] = 0.0
                # print(lag_indices.shape)
                # print(lag_indices[:500, :])
            # Lag features created -------------

        # print(X.head())
        X = X[[client+'_Prev'] + ['Hour', 'DayOfWeek', 'tsId']]
        X = X.values.tolist()
        if params.isLagFeat:
            lagX = lagX.tolist()
        Y = Y.values.tolist()
        XX.append(X)
        if params.isLagFeat:
            lagXX.append(lagX)
        #print(lagXX[-1])
        YY.append(Y)

    with open('datasets/electricity.pkl','wb') as f:
        pickle.dump(XX, f)
        pickle.dump(YY, f)

    if params.isLagFeat:
        with open('datasets/electricity_lag_feats.pkl','wb') as f:
            pickle.dump(lagXX, f)


def getEmbeddingElectricity(X):
    feature_ind_dict = OrderedDict([('prev_y', 0),
                                    ('hour', 1),
                                    ('dayofweek', 2),
                                    ('ts', 3),
                                    ])
    emb_dict = OrderedDict([#('minute', (4, 2)),
                            #('hour', (24, 9)),
                            #('day', (31, 10)),
                            #('month', (12, 6)),
                            #('year', (5, 3)),
                            #('dayofweek', (7, 4)),
                            ('ts', (370, 20)),
                            ])

    structure = []

    for feature, idx in feature_ind_dict.items():
        if feature not in emb_dict.keys():
            structure.append(tf.cast(X[:, :, idx:idx+1], tf.float32))
        else:
            num_vals, emb_size = emb_dict[feature]
            v_index = tf.Variable(tf.random_uniform([num_vals, emb_size], -1.0, 1.0, seed=12))
            em_index = tf.nn.embedding_lookup(v_index, tf.cast(X[:, :, idx], tf.int32))
            structure.append(em_index)

    X_embd = tf.concat(structure, axis=2)
    return X_embd


def main():
    createPickleFromRawData(params, writeToFile=True)

if __name__ == '__main__':
    main()
