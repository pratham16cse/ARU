import carpartsUtil
import weekly_sales_util
import walmart_sales_utils
import RossmanUtils
import synthetic_utils
import oneC_utils
import electricity_utils
import trafficPEMS_utils
import tourism_utils
import cnp_synthetic_utils
import pickle
import numpy as np
from collections import  OrderedDict
np.random.seed(12)

def readAndPickleRawDataset(params):
    if params.dataset == "carparts":
    	#XXTrain,YYTrain,PrevYYTrain,XXTest,YYTest,PrevYYTest,count,numFW = carpartsUtil.getValues(params.encoder_length, params.decoder_length, params.testFraction)
    	#maxYY = []
    	#isNormalised = False
        XX, lagXX, YY, feature_dict, emb_dict = carpartsUtil.createPickleFromRawData(params, writeToFile=False)
    elif params.dataset == "synthetic":
        XX, lagXX, YY, feature_dict, emb_dict = synthetic_utils.createPickleFromRawData(params, writeToFile=False)
    elif params.dataset == "weeklySales":
        XX, lagXX, YY, feature_dict, emb_dict = weekly_sales_util.createPickleFromRawData()
    elif params.dataset == "walmart":
    	XX, lagXX, YY, feature_dict, emb_dict = walmart_sales_utils.createPickleFromRawData(params, writeToFile=False)
    elif params.dataset == "rossman":
    	XX, lagXX, YY, feature_dict, emb_dict = RossmanUtils.createPickleFromRawData(params, writeToFile=False)
    elif params.dataset =='oneC':
        oneC_utils.createPickleFromRawData()
    elif params.dataset == 'electricity':
        XX, lagXX, YY, feature_dict, emb_dict = electricity_utils.createPickleFromRawData(params, writeToFile=False)
    elif params.dataset == 'trafficPEMS':
        XX, lagXX, YY, feature_dict, emb_dict = trafficPEMS_utils.createPickleFromRawData(params, writeToFile=False)
    elif params.dataset == 'tourism':
        XX, lagXX, YY, feature_dict, emb_dict = tourism_utils.createPickleFromRawData(params, writeToFile=False)
    elif params.dataset == 'cnp_synthetic':
        XX, lagXX, YY, feature_dict, emb_dict = tourism_utils.createPickleFromRawData(params, writeToFile=False)

    return XX, lagXX, YY, feature_dict, emb_dict 


def readFromPickle(params):
    sequence_length = params.encoder_length + params.decoder_length
    XX, lagXX, YY, feature_dict, emb_dict = readAndPickleRawDataset(params)
#    with open('datasets/'+params.dataset+'.pkl', 'r') as f:
#        XX = pickle.load(f)
#        YY = pickle.load(f)
#    with open('datasets/'+params.dataset+'_lag_feats.pkl', 'r') as f:
#        lagXX = pickle.load(f)

    XTrain, lagXTrain, XValidation, lagXValidation, XTest, lagXTest, YTrain, YValidation, YTest, maxYY, avgYY = list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list()

    norm_mask = np.zeros((len(feature_dict.keys()),), dtype='bool')
    for feature, ind in feature_dict.items():
        if feature in emb_dict.keys():
            norm_mask[ind] = True

    assert params.valFrom in ['train', 'test']
    if params.valFrom == 'train':
        #val_indices = sorted(np.random.randint(low=0, high=len(XX)-1, size=int(len(XX)/10)).tolist())
        val_indices = range(len(XX))
    elif params.valFrom == 'test':
        val_indices = []
    #print(sorted(val_indices))
    print('No of timeseries:', len(XX))
    indices = range(len(XX))
    #print(indices)
    oldind2nedind = OrderedDict()
    for ind, X, lagX, y in zip(indices, XX, lagXX, YY):

        test_end = len(y)
        test_start = test_end - params.test_length - params.encoder_length
        if ind in val_indices:
            val_end = test_end - params.test_length
            val_start = val_end - params.val_length - params.encoder_length
            train_end = val_end - params.val_length
        else:
            val_end = None
            val_start = None
            train_end = test_end - params.test_length
        train_start = 0
        test_length = test_end - test_start
        if ind in val_indices:
            val_length = val_end - val_start
        train_length = train_end - train_start

#        test_end = len(y)
#        test_start = test_end - params.test_length - params.encoder_length
#        if ind in val_indices:
#            val_end = test_start
#            val_start = val_end - sequence_length
#            train_end = val_start
#        else:
#            val_end = None
#            val_start = None
#            train_end = test_start
#        train_start = 0
#        test_length = test_end - test_start
#        if ind in val_indices:
#            val_length = val_end - val_start
#        else:
#            val_length = None
#        train_length = train_end - train_start


        # if params.verbose:
        #     print(train_start, train_end, test_start, test_end)
        #     print(train_length, test_length)
        #     print(len(y), 2 * params.decoder_length + train_length + 1)

        if ind in val_indices:
            threshold = 2 * params.decoder_length + train_length
        else:
            threshold = params.decoder_length + train_length
        if len(y) >= threshold and train_length >= sequence_length:
            XTr = np.array(X[train_start:train_end])
            lagXTr = np.array(lagX[train_start:train_end])
            YTr = np.array(y[train_start:train_end])
            if ind in val_indices:
                XVal = np.array(X[val_start:val_end])
                lagXVal = np.array(lagX[val_start:val_end])
                YVal = np.array(y[val_start:val_end])
            XTe = np.array(X[test_start:test_end])
            lagXTe = np.array(lagX[test_start:test_end])
            YTe = np.array(y[test_start:test_end])

            if params.isLogNormalised:
                assert np.sum(XTr[:, 0] >= 0) == XTr.shape[0] * XTr.shape[1]
                XTr[:, 0] = np.log(XTr[:, 0]) # Log Normalize x train
                XTr[np.isnan(XTr[:, 0]), 0] = 0
                XTr[np.isinf(XTr[:, 0]), 0] = 0

                assert np.sum(lagXTr[:, 0] >= 0) == lagXTr.shape[0] * lagXTr.shape[1]
                lagXTr[:, 0] = np.log(lagXTr[:, 0]) # Log Normalize lag-x train
                lagXTr[np.isnan(lagXTr[:, 0]), 0] = 0
                lagXTr[np.isinf(lagXTr[:, 0]), 0] = 0

                if ind in val_indices:
                    assert np.sum(XVal[:, 0] >= 0) == XVal.shape[0] * XVal.shape[1]
                    XVal[:, 0] = np.log(XVal[:, 0]) # Log Normalize x val
                    XVal[np.isnan(XVal[:, 0]), 0] = 0
                    XVal[np.isinf(XVal[:, 0]), 0] = 0

                    assert np.sum(lagXVal[:, 0] >= 0) == lagXVal.shape[0] * lagXVal.shape[1]
                    lagXVal[:, 0] = np.log(lagXVal[:, 0]) # Log Normalize lag-x val
                    lagXVal[np.isnan(lagXVal[:, 0]), 0] = 0
                    lagXVal[np.isinf(lagXVal[:, 0]), 0] = 0

                assert np.sum(XTe[:, 0] >= 0) == XTe.shape[0] * XTe.shape[1]
                XTe[:, 0] = np.log(XTe[:, 0]) # Log Normalize x test
                XTe[np.isnan(XTe[:, 0]), 0] = 0
                XTe[np.isinf(XTe[:, 0]), 0] = 0

                assert np.sum(lagXTe[:, 0] >= 0) == lagXTe.shape[0] * lagXTe.shape[1]
                lagXTe[:, 0] = np.log(lagXTe[:, 0]) # Log Normalize lag-x val
                lagXTe[np.isnan(lagXTe[:, 0]), 0] = 0
                lagXTe[np.isinf(lagXTe[:, 0]), 0] = 0

                assert np.sum(YTr >= 0) == YTr.shape[0] * YTr.shape[1]
                YTr = np.log(YTr) # Log Normalize y train
                YTr[np.isnan(YTr)] = 0
                YTr[np.isinf(YTr)] = 0

                if ind in val_indices:
                    assert np.sum(YVal >= 0) == YVal.shape[0] * YVal.shape[1]
                    YVal = np.log(YVal) # Log Normalize y val
                    YVal[np.isnan(YVal)] = 0
                    YVal[np.isinf(YVal)] = 0

                assert np.sum(YTe >= 0) == YTe.shape[0] * YTe.shape[1]
                YTe = np.log(YTe) # Log Normalize y test
                YTe[np.isnan(YTe)] = 0
                YTe[np.isinf(YTe)] = 0
            # ---- Log Scaling Done ---- #

            if params.deep_ar_normalize_seq == True or params.isNormalised == True:
                if params.deep_ar_normalize_seq == True:
                    avgY = 1 + YTr.mean()
                    YTr = YTr/avgY # Normalize y train
                    YTr[np.isnan(YTr)] = 0
                    YTr[np.isinf(YTr)] = 0
                    avgYY.append([avgY]) # For denormalization
                    if ind in val_indices:
                        YVal = YVal/avgY # Normalize y val
                        YVal[np.isnan(YVal)] = 0
                        YVal[np.isinf(YVal)] = 0
                    YTe = YTe/avgY # Normalize y test
                    YTe[np.isnan(YTe)] = 0
                    YTe[np.isinf(YTe)] = 0
                else:
                    avgY = 1 + YTr.mean()
                    avgYY.append([avgY])
                # ---- deep-ar-normalization Normalization Done ---- #

                if params.isNormalised == True:
                    maxY = YTr.max()
                    YTr = YTr/maxY # Normalize y train
                    YTr[np.isnan(YTr)] = 0
                    YTr[np.isinf(YTr)] = 0
                    maxYY.append([maxY]) # For denormalization
                    if ind in val_indices:
                        YVal = YVal/maxY # Normalize y val
                        YVal[np.isnan(YVal)] = 0
                        YVal[np.isinf(YVal)] = 0
                    YTe = YTe/maxY # Normalize y test
                    YTe[np.isnan(YTe)] = 0
                    YTe[np.isinf(YTe)] = 0
                else:
                    maxY = YTr.max()
                    maxYY.append([maxY])

                # ---- Divide-by-max Normalization Done ---- #


#                maxX = XTr.max(axis=0)
#                maxX[norm_mask] = 1.0
#                XTr = XTr/maxX # Normalize x train
#                XTr[np.isnan(XTr)] = 0
#                XTr[np.isinf(XTr)] = 0
#                if ind in val_indices:
#                    XVal = XVal/maxX # Normalize x val
#                    XVal[np.isnan(XVal)] = 0
#                    XVal[np.isinf(XVal)] = 0
#                XTe = XTe/maxX # Normalize x test
#                XTe[np.isnan(XTe)] = 0
#                XTe[np.isinf(XTe)] = 0

                if params.deep_ar_normalize_seq:
                    lagXTr = lagXTr/avgY # Normalize x train
                    if ind in val_indices:
                        lagXVal = lagXVal/avgY # Normalize x val
                    lagXTe = lagXTe/avgY # Normalize x test
                elif params.isNormalised:
                    lagXTr = lagXTr/maxY # Normalize x train
                    if ind in val_indices:
                        lagXVal = lagXVal/maxY # Normalize x val
                    lagXTe = lagXTe/maxY # Normalize x test
                lagXTr[np.isnan(lagXTr)] = 0
                lagXTr[np.isinf(lagXTr)] = 0
                if ind in val_indices:
                    lagXVal[np.isnan(lagXVal)] = 0
                    lagXVal[np.isinf(lagXVal)] = 0
                lagXTe[np.isnan(lagXTe)] = 0
                lagXTe[np.isinf(lagXTe)] = 0

                # ---- Divide-by-max Normalization On FEATURES Done---- #
            else:
                avgY = 1 + YTr.mean()
                avgYY.append([avgY])
                maxY = YTr.max()
                maxYY.append([maxY])

            XTrain.append(XTr.tolist())
            lagXTrain.append(lagXTr.tolist())
            YTrain.append(YTr.tolist())
            if ind in val_indices:
                XValidation.append(XVal.tolist())
                lagXValidation.append(lagXVal.tolist())
                YValidation.append(YVal.tolist())
                oldind2nedind[ind] = len(XTrain)-1
            XTest.append(XTe.tolist())
            lagXTest.append(lagXTe.tolist())
            YTest.append(YTe.tolist())

    assert len(YTrain) == len(avgYY)
    assert len(YTrain) == len(maxYY)

    if params.verbose:
        print('Number of Timeseries (Train): '+str(len(XTrain)))
        print('Length of first TS: '+str(len(XTrain[0])))
        if len(XValidation):
            print('Number of Timeseries (Validation): '+str(len(XValidation))+', Length of first TS: '+str(len(XValidation[0])))
        print('Number of Timeseries (Test): '+str(len(XTest))+', Length of first TS: '+str(len(XTest[0])))

    val_indices_new = list()
    for old_ind, new_ind in oldind2nedind.items():
        if old_ind in val_indices:
            val_indices_new.append(new_ind)

    return XTrain, lagXTrain, YTrain, XValidation, lagXValidation, YValidation, XTest, lagXTest, YTest, maxYY, avgYY, val_indices_new, feature_dict, emb_dict
