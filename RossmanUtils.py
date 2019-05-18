from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import pandas as pd
import re
import numpy as np
import tensorflow as tf
import math
import pandas as pd

import collections
from collections import OrderedDict

from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

import pickle
import pickle
from prepare_nn_features import CompetitionOpenSinceYear2int
from sklearn.preprocessing import StandardScaler


# In[4]:


def getActualFeatures():
    features = ['store_open', 'store_index', 'day_of_week', 
            'promo', 'year', 'month', 'day', 'state_holiday', 
            'school_holiday', 'has_competition_for_months', 
            'has_promo2_for_weeks', 'latest_promo2_for_months', 
            'distance', 'StoreType', 'Assortment', 'PromoInterval', 
            'CompetitionOpenSinceYear', 'Promo2SinceYear', 'State', 
            'week_of_year','weather1_temperature','weather2_temperature','weather3_temperature',
            'weather4_humidity','weather5_humidity','weather6_humidity',
            'weather7_wind','weather8_wind','weather9_cloud','weather10_weather_event',
            'fb1_promo_first_forward_looking','fb2_promo_last_backward_looking',
            'fb3_stateHoliday_first_forward_looking','fb4_stateHoliday_last_backward_looking',
            'fb5_stateHoliday_count_forward_looking','fb6_stateHoliday_count_backward_looking',
            'fb7_schoolHoliday_first_forward_looking','fb8_schoolHoliday_last_backward_looking',
            'google_trend_DE','google_trend_state'
           ]
    return features

def getFeatures():
    features = ['store_open', 'store_index', 'day_of_week', 
            'promo', 'year', 'month', 'day', 'state_holiday', 
            'school_holiday', 'has_competition_for_months', 
            'has_promo2_for_weeks', 'latest_promo2_for_months', 
            'distance', 'StoreType', 'Assortment', 'PromoInterval', 
            'CompetitionOpenSinceYear', 'Promo2SinceYear', 'State', 
            'week_of_year','weather1_temperature','weather2_temperature','weather3_temperature',
            'weather4_humidity','weather5_humidity','weather6_humidity',
            'weather7_wind','weather8_wind','weather9_cloud','weather10_weather_event',
            'fb1_promo_first_forward_looking','fb2_promo_last_backward_looking',
            'fb3_stateHoliday_first_forward_looking','fb4_stateHoliday_last_backward_looking',
            'fb5_stateHoliday_count_forward_looking','fb6_stateHoliday_count_backward_looking',
            'fb7_schoolHoliday_first_forward_looking','fb8_schoolHoliday_last_backward_looking',
            'google_trend_DE','google_trend_state'
           ]
    return features

def getNumericalFeatures():
    features = ['promo', 'school_holiday',
            'distance', 'weather1_temperature', 'weather2_temperature', 'weather3_temperature',
            'weather4_humidity', 'weather5_humidity', 'weather6_humidity', 'weather7_wind', 'weather8_wind', 'weather9_cloud', 'google_trend_DE', 'google_trend_state'
           ]
    return features

# In[5]:


def getTrainTestData():
    f = open('Rossman/feature_train_data.pickle', 'rb')
    (X, y) = pickle.load(f)
#    f_Test = open('Rossman/feature_test_data.pickle', 'rb')
#    (X_Test) = pickle.load(f_Test)
    features = getFeatures()
    df = pd.DataFrame.from_records(X,columns=features)
    df.drop(columns=['store_open'])
    df['SalesAct'] = y
#    dfTest = pd.DataFrame.from_records(X_Test,columns=features)
#    dfTest = dfTest[features]
    
    def categoriseFeatures(dataF):
        dataF['store_index'] = dataF['store_index']-1
        dataF['day_of_week'] = dataF['day_of_week']-1
        #dataF['year'] = dataF['year']-2013
        #dataF['month'] = dataF['month']-1
        #dataF['day'] = dataF['day']-1
        dataF['StoreType'] -= 1
        dataF['Assortment'] -= 1
        dataF['Promo2SinceYear'] = dataF['Promo2SinceYear']-2008
        dataF['week_of_year'] = dataF['week_of_year']-1
        dataF['fb1_promo_first_forward_looking'] = dataF['fb1_promo_first_forward_looking']-1
        dataF['fb2_promo_last_backward_looking'] = dataF['fb2_promo_last_backward_looking']-1
        dataF['fb3_stateHoliday_first_forward_looking'] = dataF['fb3_stateHoliday_first_forward_looking']-1
        dataF['fb4_stateHoliday_last_backward_looking'] = dataF['fb4_stateHoliday_last_backward_looking']-1
        dataF['fb7_schoolHoliday_first_forward_looking'] = dataF['fb7_schoolHoliday_first_forward_looking']-1
        dataF['fb8_schoolHoliday_last_backward_looking'] = dataF['fb8_schoolHoliday_last_backward_looking']-1
        dataF['CompetitionOpenSinceYear'] = CompetitionOpenSinceYear2int(dataF['CompetitionOpenSinceYear'].values)-1
        cat_col = ['Promo2SinceYear']
        vals = np.sort(dataF['Promo2SinceYear'].unique())
        for c, v in enumerate(vals):
            dataF.loc[dataF['Promo2SinceYear'] == v, 'Promo2SinceYear'] = c
        return dataF

    df = categoriseFeatures(df)
#    dfTest = categoriseFeatures(dfTest)
    gt_de_enc = StandardScaler()
    gt_de_enc.fit(df['google_trend_DE'].values.reshape((-1,1)))
    gt_state_enc = StandardScaler()
    gt_state_enc.fit(df['google_trend_state'].values.reshape((-1,1)))
    df['google_trend_DE'] = gt_de_enc.transform(df['google_trend_DE'].values.reshape((-1,1))).reshape((1,-1))[0]
    df['google_trend_state'] = gt_state_enc.transform(df['google_trend_state'].values.reshape((-1,1))).reshape((1,-1))[0]
#    dfTest['google_trend_DE'] = gt_de_enc.transform(dfTest['google_trend_DE'].values.reshape((-1,1))).reshape((1,-1))[0]
#    dfTest['google_trend_state'] = gt_state_enc.transform(dfTest['google_trend_state'].values.reshape((-1,1))).reshape((1,-1))[0]
    
    return df

def createPickleFromRawData(params, writeToFile=False):

    df = getTrainTestData()
    df['Date'] = pd.to_datetime(df[['year','month','day']])
    df = df.drop(columns=['store_open', 'year'])
    df['month'] -= 1
    df['day'] -= 1
    numerical_features = getNumericalFeatures()
    df[numerical_features] /= df[numerical_features].max()
    df = df.sort_values(['store_index','Date'])
    #features = df.columns.tolist()
    df = df.set_index(['store_index','Date'])

    # Create individual timeseries from dataframe
    XX = list()
    lagXX = list()
    YY = list()
    for tsId, (store, ts) in enumerate(df.groupby(level='store_index')):
        #print(tsId, store, ts.shape)
        ts = ts.reset_index(level=['store_index'], drop=True) # Date is the new index now.
        if params.isTsId:
            ts['tsId'] = tsId

        Y = ts[['SalesAct']].values.tolist()
        YY.append(Y)

        lags = [{'days':1}]
        if params.isLagFeat:
            lags += [{'weeks':52}]
        lagX = list()
        for lag in lags:
            lagX.append(ts['SalesAct'].get(ts.index - pd.DateOffset(**lag)).fillna(0).reset_index(drop=True))
        lagX = pd.concat(lagX, axis=1, ignore_index=True)
        lagX = lagX.values.tolist()
        lagXX.append(lagX)

        ts = ts.drop(columns='SalesAct')
        X = ts.values.tolist()
        XX.append(X)

        features = ts.columns.values

    feature_dict = OrderedDict([(x,i) for i,x in enumerate(features)])
    emb_list = get_emb_list()
    if params.isTsId:
        emb_list += [('tsId', (len(XX), 50))]
    emb_dict = OrderedDict(emb_list)

    if writeToFile:
        with open('datasets/rossman.pkl','wb') as f:
            pickle.dump(XX, f)
            pickle.dump(lagXX, f)
            pickle.dump(YY, f)
            pickle.dump(feature_dict, f)
            pickle.dump(emb_dict, f)

    return XX, lagXX, YY, feature_dict, emb_dict


def createPickleFromRawData_old():

    df,dfTest,X_Test,salesLogMax,salesLogMin = getTrainTestData()

    xF = getFeatures()

    yF = ['SalesAct']

    df = df.sort_values(['store_index','year','month','day'])

    maxC = 250
    XX = list()
    YY = list()
    count = 1115

    for i in range(count):
        store_df = df.loc[df['store_index'] == i]
        valsX = store_df[xF]
        valsY = store_df[yF]
        #print(store_df[yF])

        X = store_df[yF+xF]
        X['SalesAct'] = X['SalesAct'].shift(1).fillna(0) # adding prev y features.
        X.rename(columns={'SalesAct':'PrevSalesAct'}, inplace=True)
        X = X.values.tolist()
        Y = store_df[yF].values.tolist()
        XX.append(X)
        YY.append(Y)

    with open('datasets/rossman.pkl', 'wb') as f:
        pickle.dump(XX, f)
        pickle.dump(YY, f)
        

def getRossmanData():
        # ---- Normalization Start ---- #
        if normalize == True:
            #if logNormalize:
            #    XTr = np.log(XTr) # Log Normalize x train
            maxX = XTr.max(axis=0)
            XTr = XTr/maxX # Normalize x train
            XTr[np.isnan(XTr)] = 0
            XTr[np.isinf(XTr)] = 0
            #if logNormalize:
            #    XTe = np.log(XTe) # Log Normalize x test
            XTe = XTe/maxX # Normalize x test
            XTe[np.isnan(XTe)] = 0
            XTe[np.isinf(XTe)] = 0
            if logNormalize:
                YTr = np.log(YTr) # Log Normalize y train
            maxY = YTr.max()
            YTr = YTr/maxY # Normalize y test
            YTr[np.isnan(YTr)] = 0
            YTr[np.isinf(YTr)] = 0
            maxYY.append(maxY) # For denormalization
            if logNormalize:
                YTe = np.log(YTe) # Log Normalize y test
            YTe = YTe/maxY # Normalize y test
            YTe[np.isnan(YTe)] = 0
            YTe[np.isinf(YTe)] = 0
        # ---- Normalization Done ---- #

        XXTr.append(XTr.tolist())
        XXTe.append(XTe.tolist())
        YYTr.append(YTr.tolist())
        YYTe.append(YTe.tolist())

    #print(XXTr[0])
    #print('----------------------------------------------------------------------------------------------')
    #print(YYTr[0])
    #print('----------------------------------------------------------------------------------------------')
    #print(maxYY)


        return XXTr,YYTr,XXTe,YYTe,count,maxYY,len(XXTr[0][0])


def get_emb_list():
    features = getFeatures()
    emb_list = list()
    #emb_list.append(('store_index', (1115,50)))
    emb_list.append(('day_of_week', (7,6)))
    emb_list.append(('month', (12, 6)))
    emb_list.append(('day', (31, 10)))
    emb_list.append(('state_holiday', (4,3)))
    emb_list.append(('has_competition_for_months', (25,2)))
    emb_list.append(('has_promo2_for_weeks', (26,4)))
    emb_list.append(('latest_promo2_for_months', (4,1)))
    emb_list.append(('StoreType', (4,2)))
    emb_list.append(('Assortment', (3,3)))
    emb_list.append(('PromoInterval', (4,3)))
    emb_list.append(('CompetitionOpenSinceYear', (17,4)))
    emb_list.append(('Promo2SinceYear', (8,4)))
    emb_list.append(('State', (12,6)))
    emb_list.append(('week_of_year', (52,10)))
    emb_list.append(('weather10_weather_event', (22,4)))
    emb_list.append(('fb1_promo_first_forward_looking', (8,1)))
    emb_list.append(('fb2_promo_last_backward_looking', (8,1)))
    emb_list.append(('fb3_stateHoliday_first_forward_looking', (8,1)))
    emb_list.append(('fb4_stateHoliday_last_backward_looking', (8,1)))
    emb_list.append(('fb5_stateHoliday_count_forward_looking', (3,1)))
    emb_list.append(('fb6_stateHoliday_count_backward_looking', (3,1)))
    emb_list.append(('fb7_schoolHoliday_first_forward_looking', (8,1)))
    emb_list.append(('fb8_schoolHoliday_last_backward_looking', (8,1)))

    return emb_list


# In[ ]:

def transformInput(input1):
    features = getFeatures()
    emb,den,inpm = getEmbeddingMap()
    
    structure = []
    
    #prevY
    prevY = tf.expand_dims(tf.cast(input1[:,:,0], tf.float32), axis=2)
    structure.append(prevY)

    #1
    #i_store_index = inpm['store_index']
    #v_store_index = tf.Variable(tf.random_uniform([emb[i_store_index][0],emb[i_store_index][1]], -1.0, 1.0))
    #em_store_index = tf.nn.embedding_lookup(v_store_index, tf.cast(input1[:,:,i_store_index:i_store_index+1],tf.int32))
    #em_store_index = tf.reshape(em_store_index,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_store_index][1]])

#     structure.append(em_store_index)

    #2
    i_day_of_week = inpm['day_of_week']
    v_day_of_week = tf.Variable(tf.random_uniform([emb[i_day_of_week][0],emb[i_day_of_week][1]], -1.0, 1.0))
    #v_day_of_week = tf.Print(v_day_of_week, [i_day_of_week, input1[:,:,i_day_of_week:i_day_of_week+1]])
    em_day_of_week = tf.nn.embedding_lookup(v_day_of_week, tf.cast(input1[:,:,i_day_of_week:i_day_of_week+1],tf.int32))
    em_day_of_week = tf.reshape(em_day_of_week,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_day_of_week][1]])

    structure.append(em_day_of_week)

    #3
    i_promo = inpm['promo']
    d_promo = tf.layers.dense(inputs=input1[:,:,i_promo:i_promo+1], units=1, activation=tf.nn.relu)

    structure.append(d_promo)


    #4
    i_year = inpm['year']
    v_year = tf.Variable(tf.random_uniform([emb[i_year][0],emb[i_year][1]], -1.0, 1.0))
    em_year = tf.nn.embedding_lookup(v_year, tf.cast(input1[:,:,i_year:i_year+1],tf.int32))
    em_year = tf.reshape(em_year,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_year][1]])

    structure.append(em_year)


    #5
    i_month = inpm['month']
    v_month = tf.Variable(tf.random_uniform([emb[i_month][0],emb[i_month][1]], -1.0, 1.0))
    em_month = tf.nn.embedding_lookup(v_month, tf.cast(input1[:,:,i_month:i_month+1],tf.int32))
    em_month = tf.reshape(em_month,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_month][1]])

    structure.append(em_month)

    #6
    i_day = inpm['day']
    v_day = tf.Variable(tf.random_uniform([emb[i_day][0],emb[i_day][1]], -1.0, 1.0))
    em_day = tf.nn.embedding_lookup(v_day, tf.cast(input1[:,:,i_day:i_day+1],tf.int32))
    em_day = tf.reshape(em_day,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_day][1]])

    structure.append(em_day)


    #7
    i_state_holiday = inpm['state_holiday']
    v_state_holiday = tf.Variable(tf.random_uniform([emb[i_state_holiday][0],emb[i_state_holiday][1]], -1.0, 1.0))
    em_state_holiday = tf.nn.embedding_lookup(v_state_holiday, tf.cast(input1[:,:,i_state_holiday:i_state_holiday+1],tf.int32))
    em_state_holiday = tf.reshape(em_state_holiday,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_state_holiday][1]])

    structure.append(em_state_holiday)


    #8
    i_school_holiday = inpm['school_holiday']
    d_school_holiday = tf.layers.dense(inputs=input1[:,:,i_school_holiday:i_school_holiday+1], units=1, activation=tf.nn.relu)

    structure.append(d_school_holiday)


    #9
    i_has_competition_for_months = inpm['has_competition_for_months']
    v_has_competition_for_months = tf.Variable(tf.random_uniform([emb[i_has_competition_for_months][0],emb[i_has_competition_for_months][1]], -1.0, 1.0))
    em_has_competition_for_months = tf.nn.embedding_lookup(v_has_competition_for_months, tf.cast(input1[:,:,i_has_competition_for_months:i_has_competition_for_months+1],tf.int32))
    em_has_competition_for_months = tf.reshape(em_has_competition_for_months,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_has_competition_for_months][1]])

    structure.append(em_has_competition_for_months)

    #10
    i_has_promo2_for_weeks = inpm['has_promo2_for_weeks']
    v_has_promo2_for_weeks = tf.Variable(tf.random_uniform([emb[i_has_promo2_for_weeks][0],emb[i_has_promo2_for_weeks][1]], -1.0, 1.0))
    em_has_promo2_for_weeks = tf.nn.embedding_lookup(v_has_promo2_for_weeks, tf.cast(input1[:,:,i_has_promo2_for_weeks:i_has_promo2_for_weeks+1],tf.int32))
    em_has_promo2_for_weeks = tf.reshape(em_has_promo2_for_weeks,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_has_promo2_for_weeks][1]])

    structure.append(em_has_promo2_for_weeks)

    #11
    i_latest_promo2_for_months = inpm['latest_promo2_for_months']
    v_latest_promo2_for_months = tf.Variable(tf.random_uniform([emb[i_latest_promo2_for_months][0],emb[i_latest_promo2_for_months][1]], -1.0, 1.0))
    em_latest_promo2_for_months = tf.nn.embedding_lookup(v_latest_promo2_for_months, tf.cast(input1[:,:,i_latest_promo2_for_months:i_latest_promo2_for_months+1],tf.int32))
    em_latest_promo2_for_months = tf.reshape(em_latest_promo2_for_months,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_latest_promo2_for_months][1]])

    structure.append(em_latest_promo2_for_months)

    #12
    i_distance = inpm['distance']
    d_distance = tf.layers.dense(inputs=input1[:,:,i_distance:i_distance+1], units=1, activation=tf.nn.relu)

    structure.append(d_distance)

    #13
    i_StoreType = inpm['StoreType']
    v_StoreType = tf.Variable(tf.random_uniform([emb[i_StoreType][0],emb[i_StoreType][1]], -1.0, 1.0))
    em_StoreType = tf.nn.embedding_lookup(v_StoreType, tf.cast(input1[:,:,i_StoreType:i_StoreType+1],tf.int32))
    em_StoreType = tf.reshape(em_StoreType,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_StoreType][1]])

    structure.append(em_StoreType)

    #14
    i_Assortment = inpm['Assortment']
    v_Assortment = tf.Variable(tf.random_uniform([emb[i_Assortment][0],emb[i_Assortment][1]], -1.0, 1.0))
    em_Assortment = tf.nn.embedding_lookup(v_Assortment, tf.cast(input1[:,:,i_Assortment:i_Assortment+1],tf.int32))
    em_Assortment = tf.reshape(em_Assortment,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_Assortment][1]])

    structure.append(em_Assortment)

    #15
    i_PromoInterval = inpm['PromoInterval']
    v_PromoInterval = tf.Variable(tf.random_uniform([emb[i_PromoInterval][0],emb[i_PromoInterval][1]], -1.0, 1.0))
    em_PromoInterval = tf.nn.embedding_lookup(v_PromoInterval, tf.cast(input1[:,:,i_PromoInterval:i_PromoInterval+1],tf.int32))
    em_PromoInterval = tf.reshape(em_PromoInterval,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_PromoInterval][1]])

    structure.append(em_PromoInterval)

    #16
    i_CompetitionOpenSinceYear = inpm['CompetitionOpenSinceYear']
    v_CompetitionOpenSinceYear = tf.Variable(tf.random_uniform([emb[i_CompetitionOpenSinceYear][0],emb[i_CompetitionOpenSinceYear][1]], -1.0, 1.0))
    em_CompetitionOpenSinceYear = tf.nn.embedding_lookup(v_CompetitionOpenSinceYear, tf.cast(input1[:,:,i_CompetitionOpenSinceYear:i_CompetitionOpenSinceYear+1],tf.int32))
    em_CompetitionOpenSinceYear = tf.reshape(em_CompetitionOpenSinceYear,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_CompetitionOpenSinceYear][1]])

    structure.append(em_CompetitionOpenSinceYear)

    #17
    i_Promo2SinceYear = inpm['Promo2SinceYear']
    v_Promo2SinceYear = tf.Variable(tf.random_uniform([emb[i_Promo2SinceYear][0],emb[i_Promo2SinceYear][1]], -1.0, 1.0))
    em_Promo2SinceYear = tf.nn.embedding_lookup(v_Promo2SinceYear, tf.cast(input1[:,:,i_Promo2SinceYear:i_Promo2SinceYear+1],tf.int32))
    em_Promo2SinceYear = tf.reshape(em_Promo2SinceYear,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_Promo2SinceYear][1]])

    structure.append(em_Promo2SinceYear)

    #18
    i_State = inpm['State']
    v_State = tf.Variable(tf.random_uniform([emb[i_State][0],emb[i_State][1]], -1.0, 1.0))
    em_State = tf.nn.embedding_lookup(v_State, tf.cast(input1[:,:,i_State:i_State+1],tf.int32))
    em_State = tf.reshape(em_State,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_State][1]])

    structure.append(em_State)

    #19
    i_week_of_year = inpm['week_of_year']
    v_week_of_year = tf.Variable(tf.random_uniform([emb[i_week_of_year][0],emb[i_week_of_year][1]], -1.0, 1.0))
    em_week_of_year = tf.nn.embedding_lookup(v_week_of_year, tf.cast(input1[:,:,i_week_of_year:i_week_of_year+1],tf.int32))
    em_week_of_year = tf.reshape(em_week_of_year,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_week_of_year][1]])

    structure.append(em_week_of_year)

    #20
    i_weather1_temperature = inpm['weather1_temperature']
    d_weather_temperature = tf.layers.dense(inputs=input1[:,:,i_weather1_temperature:i_weather1_temperature+3], units=3, activation=tf.nn.relu)

    structure.append(d_weather_temperature)

    #23
    i_weather4_humidity = inpm['weather4_humidity']
    d_weather_humidity = tf.layers.dense(inputs=input1[:,:,i_weather4_humidity:i_weather4_humidity+3], units=3, activation=tf.nn.relu)

    structure.append(d_weather_humidity)

    #26
    i_weather7_wind = inpm['weather7_wind']
    d_weather_wind = tf.layers.dense(inputs=input1[:,:,i_weather7_wind:i_weather7_wind+2], units=2, activation=tf.nn.relu)

    structure.append(d_weather_wind)

    #28
    i_weather9_cloud = inpm['weather9_cloud']
    d_weather_cloud = tf.layers.dense(inputs=input1[:,:,i_weather9_cloud:i_weather9_cloud+1], units=1, activation=tf.nn.relu)

    structure.append(d_weather_cloud)

    #29
    i_weather_event = inpm['weather10_weather_event']
    v_weather_event = tf.Variable(tf.random_uniform([emb[i_weather_event][0],emb[i_weather_event][1]], -1.0, 1.0))
    em_weather_event = tf.nn.embedding_lookup(v_weather_event, tf.cast(input1[:,:,i_weather_event:i_weather_event+1],tf.int32))
    em_weather_event = tf.reshape(em_weather_event,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_weather_event][1]])

    structure.append(em_weather_event)

    #30
    i_promo_first_forward_looking = inpm['fb1_promo_first_forward_looking']
    v_promo_first_forward_looking = tf.Variable(tf.random_uniform([emb[i_promo_first_forward_looking][0],emb[i_promo_first_forward_looking][1]], -1.0, 1.0))
    em_promo_first_forward_looking = tf.nn.embedding_lookup(v_promo_first_forward_looking, tf.cast(input1[:,:,i_promo_first_forward_looking:i_promo_first_forward_looking+1],tf.int32))
    em_promo_first_forward_looking = tf.reshape(em_promo_first_forward_looking,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_promo_first_forward_looking][1]])

    structure.append(em_promo_first_forward_looking)

    #31
    i_promo_last_backward_looking = inpm['fb2_promo_last_backward_looking']
    v_promo_last_backward_looking = tf.Variable(tf.random_uniform([emb[i_promo_last_backward_looking][0],emb[i_promo_last_backward_looking][1]], -1.0, 1.0))
    em_promo_last_backward_looking = tf.nn.embedding_lookup(v_promo_last_backward_looking, tf.cast(input1[:,:,i_promo_last_backward_looking:i_promo_last_backward_looking+1],tf.int32))
    em_promo_last_backward_looking = tf.reshape(em_promo_last_backward_looking,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_promo_last_backward_looking][1]])

    structure.append(em_promo_last_backward_looking)

    #32
    i_stateHoliday_first_forward_looking = inpm['fb3_stateHoliday_first_forward_looking']
    v_stateHoliday_first_forward_looking = tf.Variable(tf.random_uniform([emb[i_stateHoliday_first_forward_looking][0],emb[i_stateHoliday_first_forward_looking][1]], -1.0, 1.0))
    em_stateHoliday_first_forward_looking = tf.nn.embedding_lookup(v_stateHoliday_first_forward_looking, tf.cast(input1[:,:,i_stateHoliday_first_forward_looking:i_stateHoliday_first_forward_looking+1],tf.int32))
    em_stateHoliday_first_forward_looking = tf.reshape(em_stateHoliday_first_forward_looking,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_stateHoliday_first_forward_looking][1]])

    structure.append(em_stateHoliday_first_forward_looking)

    #33
    i_stateHoliday_last_backward_looking = inpm['fb4_stateHoliday_last_backward_looking']
    v_stateHoliday_last_backward_looking = tf.Variable(tf.random_uniform([emb[i_stateHoliday_last_backward_looking][0],emb[i_stateHoliday_last_backward_looking][1]], -1.0, 1.0))
    em_stateHoliday_last_backward_looking = tf.nn.embedding_lookup(v_stateHoliday_last_backward_looking, tf.cast(input1[:,:,i_stateHoliday_last_backward_looking:i_stateHoliday_last_backward_looking+1],tf.int32))
    em_stateHoliday_last_backward_looking = tf.reshape(em_stateHoliday_last_backward_looking,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_stateHoliday_last_backward_looking][1]])

    structure.append(em_stateHoliday_last_backward_looking)

    #34
    i_stateHoliday_count_forward_looking = inpm['fb5_stateHoliday_count_forward_looking']
    v_stateHoliday_count_forward_looking = tf.Variable(tf.random_uniform([emb[i_stateHoliday_count_forward_looking][0],emb[i_stateHoliday_count_forward_looking][1]], -1.0, 1.0))
    em_stateHoliday_count_forward_looking = tf.nn.embedding_lookup(v_stateHoliday_count_forward_looking, tf.cast(input1[:,:,i_stateHoliday_count_forward_looking],tf.int32))
    em_stateHoliday_count_forward_looking = tf.reshape(em_stateHoliday_count_forward_looking,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_stateHoliday_count_forward_looking][1]])

    structure.append(em_stateHoliday_count_forward_looking)


    #35
    i_stateHoliday_count_backward_looking = inpm['fb6_stateHoliday_count_backward_looking']
    v_stateHoliday_count_backward_looking = tf.Variable(tf.random_uniform([emb[i_stateHoliday_count_backward_looking][0],emb[i_stateHoliday_count_backward_looking][1]], -1.0, 1.0))
    em_stateHoliday_count_backward_looking = tf.nn.embedding_lookup(v_stateHoliday_count_backward_looking, tf.cast(input1[:,:,i_stateHoliday_count_backward_looking:i_stateHoliday_count_backward_looking+1],tf.int32))
    em_stateHoliday_count_backward_looking = tf.reshape(em_stateHoliday_count_backward_looking,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_stateHoliday_count_backward_looking][1]])

    structure.append(em_stateHoliday_count_backward_looking)

    #36
    i_schoolHoliday_first_forward_looking = inpm['fb7_schoolHoliday_first_forward_looking']
    v_schoolHoliday_first_forward_looking = tf.Variable(tf.random_uniform([emb[i_schoolHoliday_first_forward_looking][0],emb[i_schoolHoliday_first_forward_looking][1]], -1.0, 1.0))
    em_schoolHoliday_first_forward_looking = tf.nn.embedding_lookup(v_schoolHoliday_first_forward_looking, tf.cast(input1[:,:,i_schoolHoliday_first_forward_looking:i_schoolHoliday_first_forward_looking+1],tf.int32))
    em_schoolHoliday_first_forward_looking = tf.reshape(em_schoolHoliday_first_forward_looking,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_schoolHoliday_first_forward_looking][1]])

    structure.append(em_schoolHoliday_first_forward_looking)

    #37
    i_schoolHoliday_last_backward_looking = inpm['fb8_schoolHoliday_last_backward_looking']
    v_schoolHoliday_last_backward_looking = tf.Variable(tf.random_uniform([emb[i_schoolHoliday_last_backward_looking][0],emb[i_schoolHoliday_last_backward_looking][1]], -1.0, 1.0))
    em_schoolHoliday_last_backward_looking = tf.nn.embedding_lookup(v_schoolHoliday_last_backward_looking, tf.cast(input1[:,:,i_schoolHoliday_last_backward_looking:i_schoolHoliday_last_backward_looking+1],tf.int32))
    em_schoolHoliday_last_backward_looking = tf.reshape(em_schoolHoliday_last_backward_looking,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_schoolHoliday_last_backward_looking][1]])

    structure.append(em_schoolHoliday_last_backward_looking)

    #38
    i_google_trend_DE = inpm['google_trend_DE']
    d_google_trend_DE = tf.layers.dense(inputs=input1[:,:,i_google_trend_DE:i_google_trend_DE+1], units=1, activation=tf.nn.relu)

    structure.append(d_google_trend_DE)

    #39
    i_google_trend_state = inpm['google_trend_state']
    d_google_trend_state = tf.layers.dense(inputs=input1[:,:,i_google_trend_state:i_google_trend_state+1], units=1, activation=tf.nn.relu)

    structure.append(d_google_trend_state)

    input11 = tf.concat(structure, axis=2)
    return input11


def main():
    testFraction = 0
    sequence_length = 16
    decoder_length = 8
    isLogNormalised = False
    isNormalised = True
    createPickleFromRawData(params, writeToFile=True)
    #XXTrain,YYTrain,XXTest,YYTest,count,maxYY,numFW,salesLogMax = getRossmanData(testFraction,sequence_length,decoder_length,isLogNormalised,isNormalised)

if __name__ == '__main__':
    main()
