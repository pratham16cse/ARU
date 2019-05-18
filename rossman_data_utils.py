from __future__ import absolute_import
from __future__ import division

import csv
import pandas as pd
import re
import numpy as np
import tensorflow as tf
import math
import pandas as pd

import collections

from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

import pickle
from prepare_nn_features import CompetitionOpenSinceYear2int
from sklearn.preprocessing import StandardScaler


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


# In[ ]:

def getTrainTestData():
    features = getFeatures()
    f = open('Rossman/feature_train_data.pickle', 'rb')
    (X, y) = pickle.load(f)
    f_Test = open('Rossman/feature_test_data.pickle', 'rb')
    (X_Test) = pickle.load(f_Test)
    featuresActual = getActualFeatures()
    features = getFeatures()
    df = pd.DataFrame.from_records(X,columns=featuresActual)
    df = df[features]
    df['SalesAct'] = y
    dfTest = pd.DataFrame.from_records(X_Test,columns=featuresActual)
    dfTest = dfTest[features]
    salesLogMax = np.log(max(df['SalesAct']))
    salesLogMin = np.log(min(df['SalesAct']))
    df['ScaledLogSales'] = (np.log(df['SalesAct'].values))/(salesLogMax)
    
    def categoriseFeatures(dataF):
        dataF[features[1]] = dataF[features[1]]-1
        dataF[features[2]] = dataF[features[2]]-1
        dataF[features[4]] = dataF[features[4]]-2013
        dataF[features[5]] = dataF[features[5]]-1
        dataF[features[6]] = dataF[features[6]]-1
        dataF[features[17]] = dataF[features[17]]-2008
        dataF[features[19]] = dataF[features[19]]-1
        dataF[features[30]] = dataF[features[30]]-1
        dataF[features[31]] = dataF[features[31]]-1
        dataF[features[32]] = dataF[features[32]]-1
        dataF[features[33]] = dataF[features[33]]-1
        dataF[features[36]] = dataF[features[36]]-1
        dataF[features[37]] = dataF[features[37]]-1
        dataF['CompetitionOpenSinceYear'] = CompetitionOpenSinceYear2int(dataF['CompetitionOpenSinceYear'].values)
        cat_col = ['Promo2SinceYear']
        for i in cat_col:
            vals = np.sort(dataF[i].unique())
            c=0
            for v in vals:
                dataF.loc[dataF[i] == v, i] = c
                c=c+1
        return dataF

    df = categoriseFeatures(df)
    dfTest = categoriseFeatures(dfTest)
    gt_de_enc = StandardScaler()
    gt_de_enc.fit(df['google_trend_DE'].values.reshape((-1,1)))
    gt_state_enc = StandardScaler()
    gt_state_enc.fit(df['google_trend_state'].values.reshape((-1,1)))
    df['google_trend_DE'] = gt_de_enc.transform(df['google_trend_DE'].values.reshape((-1,1))).reshape((1,-1))[0]
    df['google_trend_state'] = gt_state_enc.transform(df['google_trend_state'].values.reshape((-1,1))).reshape((1,-1))[0]
    dfTest['google_trend_DE'] = gt_de_enc.transform(dfTest['google_trend_DE'].values.reshape((-1,1))).reshape((1,-1))[0]
    dfTest['google_trend_state'] = gt_state_enc.transform(dfTest['google_trend_state'].values.reshape((-1,1))).reshape((1,-1))[0]
    
    return df,dfTest,X_Test,salesLogMax,salesLogMin


# In[ ]:

def getDataForIndex(dataF,i_index):
    dfS = dataF.loc[dataF['store_index'] == i_index]
    return dfS

def getEmbeddingMap():
    features = getFeatures()
    emb = {}
    den = {}
    inpm = {}
    emb[1] = (1115,50)
    emb[2] = (7,6)
    den[3] = (1,1)
    emb[4] = (3,2)
    emb[5] = (12,6)
    emb[6] = (31,10)
    emb[7] = (4,3)
    den[8] = (1,1)
    emb[9] = (25,2)
    emb[10] = (26,1)
    emb[11] = (4,1)
    den[12] = (1,1)
    emb[13] = (5,2)
    emb[14] = (4,3)
    emb[15] = (4,3)
    emb[16] = (18,4)
    emb[17] = (8,4)
    emb[18] = (12,6)
    emb[19] = (53,2)
    den[22] = (3,3)
    den[25] = (3,3)
    den[27] = (2,2)
    den[28] = (1,1)
    emb[29] = (22,4)
    emb[30] = (8,1)
    emb[31] = (8,1)
    emb[32] = (8,1)
    emb[33] = (8,1)
    emb[34] = (3,1)
    emb[35] = (3,1)
    emb[36] = (8,1)
    emb[37] = (8,1)
    den[38] = (1,1)
    den[39] = (1,1)
    for f in range(len(features)):
        inpm[features[f]] = f
    
    return emb,den,inpm


# In[ ]:

def transformInput(input1):
    features = getFeatures()
    emb,den,inpm = getEmbeddingMap()
    
    structure = []
    #1
    i_store_index = inpm['store_index']
    v_store_index = tf.Variable(tf.random_uniform([emb[i_store_index][0],emb[i_store_index][1]], -1.0, 1.0))
    em_store_index = tf.nn.embedding_lookup(v_store_index, tf.cast(input1[:,:,i_store_index:i_store_index+1],tf.int32))
    em_store_index = tf.reshape(em_store_index,[tf.shape(input1)[0],tf.shape(input1)[1],emb[i_store_index][1]])

#     structure.append(em_store_index)

    #2
    i_day_of_week = inpm['day_of_week']
    v_day_of_week = tf.Variable(tf.random_uniform([emb[i_day_of_week][0],emb[i_day_of_week][1]], -1.0, 1.0))
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
    em_stateHoliday_count_forward_looking = tf.nn.embedding_lookup(v_stateHoliday_count_forward_looking, tf.cast(input1[:,:,i_stateHoliday_count_forward_looking:i_stateHoliday_count_forward_looking+1],tf.int32))
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
    return structure