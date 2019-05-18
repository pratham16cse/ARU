import pandas as pd
import numpy as np
import datetime
import pickle
import tensorflow as tf

def dateBlockToMY(dbn):
    # Returns month and year given date_block_num
    return 2013+int(dbn)/12, int(dbn)%12+1

def createPickleFromRawData():
    sales = pd.read_csv('Predict_Future_Sales/sales_train.csv')
    item_categories = pd.read_csv('Predict_Future_Sales/item_categories.csv')
    items = pd.read_csv('Predict_Future_Sales/items.csv')
    shops = pd.read_csv('Predict_Future_Sales/shops.csv')
    subset = ['date','date_block_num','shop_id','item_id','item_cnt_day']
    sales.drop_duplicates(subset=subset, inplace=True)
    sales.date = sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
    monthly_sales=sales.groupby(['date_block_num','shop_id','item_id'])[
            'date','item_price','item_cnt_day'].agg({'date':'min','item_price':'mean','item_cnt_day':'sum'})
    monthly_sales['month'] = monthly_sales.date.apply(lambda x: int(x.strftime('%m')))
    monthly_sales['year'] = monthly_sales.date.apply(lambda x: int(x.strftime('%Y')))
    monthly_sales = monthly_sales.drop('date', axis=1)
    monthly_sales = monthly_sales.reset_index()
    all_ts = monthly_sales.reset_index().groupby(['shop_id','item_id'])
    
    #print(monthly_sales.head(20))
    #print(monthly_sales.iloc[all_ts.groups[(2,4271)]])

    pair2id = dict()
    for item, shop in monthly_sales[['item_id','shop_id']].values:
        if pair2id.get((item, shop), -1) == -1:
            pair2id[(item, shop)] = 1
        else:
            pair2id[(item, shop)] += 1

    XX = list()
    YY = list()
    #XTest = list()
    #YTest = list()
    #maxY = list()
    xF = ['item_price','month','year']
    yF = ['item_cnt_day']

    for cnt, ((item, shop), sales) in enumerate(pair2id.items()):
        if sales >= 20: # Select only (item, shop) pairs with >=20 sales across training duration
            print(cnt, item, shop)
            M = monthly_sales.loc[(monthly_sales['item_id'] == item) & (monthly_sales['shop_id'] == shop)].reset_index()
            M = M.drop('index', axis=1)
            date_blocks = list(sorted(set(M['date_block_num'].values)))
            missing_date_blocks = list(sorted(set(range(34)) - set(date_blocks)))
            r, c = M.shape
            #if len(missing_date_blocks):
            #    mdbn = missing_date_blocks[0]
            #    M.loc[r] = [mdbn, shop, item,  M.loc[M['date_block_num']==date_blocks[0]]['item_price'].values[0], 0, dateBlockToMY(mdbn)[1], dateBlockToMY(mdbn)[0]]
            #    date_blocks = [mdbn] + date_blocks
            #    for i, mdbn in enumerate(missing_date_blocks[1:]):
            #        M.loc[r+i+1] = [mdbn, shop, item,  M.loc[M['date_block_num']==mdbn-1]['item_price'].values[0], 0, dateBlockToMY(mdbn)[1], dateBlockToMY(mdbn)[0]]

            M = M.sort_values(['date_block_num'])

            X = M[yF+xF]
            X['item_cnt_day'] = X['item_cnt_day'].shift(1).fillna(0) # add previous y feature.
            X.rename(columns={'item_cnt_day':'prev_item_cnt'}, inplace=True)
            X['year'] = X['year'] - 2013
            X['month'] = X['month'] - 1
            X = X.values.tolist()
            Y = M[yF].values.tolist()

            XX.append(X)
            YY.append(Y)

    #print(XTrain[0])
    with open('datasets/oneC.pkl','wb') as f:
        pickle.dump(XX, f)
        pickle.dump(YY, f)

def getEmbeddingOneC(X):
    month_embed_size = 6 #12
    year_embed_size = 1 #3

    structure = []

    #prev Y and item_price
    prevY = tf.cast(X[:,:,0:2], tf.float32)
    structure.append(prevY)
    
    #month
    v_month_index = tf.Variable(tf.random_uniform([12,month_embed_size], -1.0, 1.0, seed=12))
    em_month_index = tf.nn.embedding_lookup(v_month_index, tf.cast(X[:,:,2],tf.int32))
    structure.append(em_month_index)

    #year
    v_year_index = tf.Variable(tf.random_uniform([3,year_embed_size], -1.0, 1.0, seed=12))
    #v_year_index = tf.Print(v_year_index, [ v_year_index, X[:,:,3]], message="HERE", summarize=30000)
    em_year_index = tf.nn.embedding_lookup(v_year_index, tf.cast(X[:,:,3],tf.int32))
    structure.append(em_year_index)
    
    X_embd = tf.concat(structure, axis=2)
    return X_embd

def getValues1C(testFraction, decoder_length, modelToRun, normalize, logNormalize):
    with open('datasets/oneC.pkl','r') as f:
        XX = pickle.load(f)
        YY = pickle.load(f)

    XTrain, XTest, YTrain, YTest, maxYY = list(), list(), list(), list(), list()
    for X, y in zip(XX, YY):
        if testFraction:
            test_length = int(testFraction*(M.shape[0]-1))
        else:
            test_length = int(decoder_length)

        XTr = np.array(X[:-test_length])
        YTr = np.array(y[:-test_length])
        XTe = np.array(X[-test_length:])
        YTe = np.array(y[-test_length:])
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

        XTrain.append(XTr.tolist())
        YTrain.append(YTr.tolist())
        XTest.append(XTe.tolist())
        YTest.append(YTe.tolist())

    return XTrain, YTrain, XTest, YTest, len(XTrain), maxYY, len(XTrain[0][0])


if __name__ == '__main__':
#    getValues1C(0,0,0,0)
    testFraction = 0
    sequence_length = 16
    decoder_length = 8
    modelToRun = 'baseline'
    normalize = True
    logNormalize = False
    createPickleFromRawData()
#    XXTrain, YYTrain, XXTest, YYTest, count, maxYY, numFW = \
#            getValues1C(testFraction, decoder_length, modelToRun, normalize, logNormalize)
    #print(len(XXTrain[0]))
    #print(len(YYTrain[0]))
#    tsId = np.random.randint(len(XXTrain))
#    X = np.array(XXTrain[tsId])
#    y = np.array(YYTrain[tsId])[:,0].tolist()
#    for i in range(X.shape[1]):
#        #pltX = X[:,i].tolist()
#        #print(pltX)
#        #print(y)
#        plt.plot(y, 'b')
#        plt.plot(y, 'k*')
#        plt.show()
