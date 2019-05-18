from __future__ import print_function

import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import OrderedDict


def parse_data(filePath):
    with open(filePath, 'r') as f:
        df = list()
        for line in f:
                ts = [[float(i) for i in row.split(' ')]
                      for row in line.strip()[1:-1].split(';')]
                ts_arr = np.array(ts)
                df.append(ts_arr)
    return df


def createPickleFromRawData(params, writeToFile=False):
    df_train = parse_data('traffic/PEMS_train')
    df_test = parse_data('traffic/PEMS_test')
    print(len(df_train), len(df_test))
    print(np.stack(df_train, axis=0).shape, np.stack(df_test, axis=0).shape)
    df = df_train + df_test # Combine both lists of timeseries
    df = np.stack(df, axis=0)
    print(df.shape)

    # Coarsening of the data - aggreate data to hourly readings
    num_readings = df.shape[2]
    df_coarse = np.zeros((df.shape[0], df.shape[1], num_readings/6))
    for idx, (i, j) in enumerate(zip(range(0, num_readings, 6), range(6, num_readings+1, 6))):
        df_coarse[:, :, idx] = np.mean(df[:,:, i:j], axis=2)
    print(df_coarse.shape)
    df = df_coarse

    randperm = parse_data('traffic/randperm')
    randperm = randperm[0][0]
    randperm = [int(i) for i in randperm]
    print(min(randperm), max(randperm))
    randperm = [i-1 for i in randperm]
    print(min(randperm), max(randperm))
    
    df_new = np.zeros_like(df)
    df_new[randperm] = df # Apply inverse permutation to get data in calendar order.
    df = df_new
    print(df.shape)

#    df_concat = df[0, :, :]
#    for i in range(1, df.shape[0]):
#        print(i)
#        df_concat =  np.concatenate((df_concat, df[i, :, :]), axis=1)
#    print(df_concat.shape)

    df_concat = np.transpose(df, [1, 0, 2]).reshape((df.shape[1], df.shape[0]*df.shape[2]))
    print(df_concat.shape)
    df = df_concat

    # Load labels
    train_labels = parse_data('traffic/PEMS_trainlabels')
    train_labels = train_labels[0][0] - 1.0
    test_labels = parse_data('traffic/PEMS_testlabels')
    test_labels = test_labels[0][0] - 1.0
    labels = train_labels.tolist() + test_labels.tolist()
    labels_new = np.zeros_like(labels)
    labels_new[randperm] = labels
    labels = labels_new

    # Create features, XX, and YY
    lagXX = np.zeros_like(df)
    lagXX[:, 1:] = df[:, :-1]
    if params.isLagFeat:
        lags = [{'weeks':1}, {'weeks':2}] # supports lags only in weeks
        df_lags = list()
        df_lags.append(lagXX)
        for lag_dict in lags:
            lag_type, lag_val = lag_dict.items()[0]
            offset = lag_val * 24 * 7
            lagX = np.pad(df, ((0,0),(offset,0)), mode='constant')[:, :-offset]
            df_lags.append(lagX)
        lagXX = np.stack(df_lags, axis=2)

    dayOfWeek = np.dot(np.expand_dims(labels, 1), np.ones((1, 24)))
    dayOfWeek = dayOfWeek.reshape((1, df.shape[1]))
    dayOfWeek = np.repeat(dayOfWeek, df.shape[0], axis=0)
    print(dayOfWeek.shape, lagXX.shape)
    XX = np.expand_dims(dayOfWeek, axis=2)
    timestamps = pd.date_range('2008-01-01','2009-04-30', freq='H') # Specified on dataset source
    timestamps = timestamps[:df.shape[1]]
    hour = np.expand_dims(np.tile(np.expand_dims(timestamps.hour.values, axis=0), [XX.shape[0], 1]), axis=2)
    day = np.expand_dims(np.tile(np.expand_dims(timestamps.day.values, axis=0), [XX.shape[0], 1]), axis=2)
    month = np.expand_dims(np.tile(np.expand_dims(timestamps.month.values, axis=0), [XX.shape[0], 1]), axis=2)
    #XX = np.concatenate([XX, day, month], axis=2)
    XX = np.concatenate([XX, hour], axis=2)
    #XX = hour

    if params.isTsId:
        tsId = np.tile(np.arange(XX.shape[0]), [XX.shape[1], 1]).T
        tsId = np.expand_dims(tsId, axis=2)
        XX = np.concatenate([XX, tsId], axis=2)
    if params.isLagFeat is not True:
        lagXX = np.expand_dims(lagXX, axis=2)
    YY = np.expand_dims(df, axis=2)

    features = ['dayOfWeek', 'hour']# 'day', 'month']
    #features = ['hour']
    #features = []
    if params.isTsId:
        features += ['tsId']
    feature_dict = OrderedDict([(x,i) for i,x in enumerate(features)])
    emb_list = [('dayOfWeek', (7, 4)), ('hour', (24, 9))]#, ('day', (31, 10)), ('month', (12, 6))]
    #emb_list = [('hour', (24, 9))]
    #emb_list = []
    if params.isTsId:
        emb_list += [('tsId', (XX.shape[0], 50))]
    emb_dict = OrderedDict(emb_list)
    print(XX.shape, YY.shape)
    if writeToFile:
        with open('datasets/trafficPEMS.pkl','wb') as f:
            pickle.dump(XX, f)
            pickle.dump(lagXX, f)
            pickle.dump(YY, f)
            pickle.dump(feature_dict, f)
            pickle.dump(emb_dict, f)

    return XX, lagXX, YY, feature_dict, emb_dict

def getEmbeddingTraffic(X):
    dayOfWeek_embed_size = 4 #7

    structure = []

    #prev Y
    prevY = tf.cast(X[:, :, 0:1], tf.float32)
    structure.append(prevY)

    # dayOfWeek
    v_dayOfWeek_index = tf.Variable(tf.random_uniform([7, dayOfWeek_embed_size], -1.0, 1.0, seed=12))
    em_dayOfWeek_index = tf.nn.embedding_lookup(v_dayOfWeek_index, tf.cast(X[:, :, 1], tf.int32))
    structure.append(em_dayOfWeek_index)

    X_embd = tf.concat(structure, axis=2)
    return X_embd


def main():
    createPickleFromRawData(params, writeToFile=False)

if __name__ == '__main__':
    main()
