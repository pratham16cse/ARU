from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function


import numpy as np
np.random.seed(12)
import tensorflow as tf
import time

import AdaptRegression as ar
import ARU as aru
import metrics
from operator import itemgetter
from models import Models
from BatchGenerator import BatchGenerator
#from models import *
#from walmart_sales_utils import *
#from AdaptRegression import *

alpha_list = [0, 0.25, 0.5, 0.9, 0.99]


def getInputPlaceHolders(modelUsed, numFW, numLagFW, sequence_length, batch_size):
    if modelUsed in ['aru', 'adapt', 'baseline', 'hybrid', 'maruY', 'maruX', 'snail', 'attention',
                     'enc_aru', 'dec_aru', 'maruX_sruAugment', 'maruX_rnnAugment', 'mEncAruY',
                     'mEncAruX_rnnAugment', 'mEncAruX_sruAugment', 'mEncAruX', 'localEncAru', 'localDecAru', 'neuralARU']:
        inputPh = tf.placeholder(tf.float32, (batch_size, sequence_length, numFW))
        lagPh = tf.placeholder(tf.float32, (batch_size, sequence_length, numLagFW))
        outputPh = tf.placeholder(tf.float32, (batch_size, sequence_length, 1))
        maxYYPh = tf.placeholder(tf.float32, (batch_size, 1))
        mode = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32)
        tsIndicesPh = tf.placeholder(tf.int32, (batch_size))
        maskPh = tf.placeholder(tf.float32, (batch_size, 1))
    elif modelUsed == "hierarchical":
        inputPh = tf.placeholder(tf.float32, (None,None,numFW))
        outputPh = tf.placeholder(tf.float32, (None,None,1))
        outputPrev = tf.placeholder(tf.float32, (None,None))
        keep_prob = tf.placeholder(tf.float32)

    return inputPh, lagPh, outputPh, maxYYPh, keep_prob, mode, tsIndicesPh, maskPh

def prependHistoryToTest(trn, tst, encoder_length):
    for i in range(len(tst)):
        tst[i] = trn[i][-encoder_length:] + tst[i]
        #tst[i] = np.concatenate((trn[i][-encoder_length:], tst[i]), axis=0)
    return tst

def prependZeros(X, lagX, y, encoder_length, numFW, numLagFW):
    for i in range(len(X)):
        X[i] = np.zeros((encoder_length , numFW)).tolist()  + X[i]
        lagX[i] = np.zeros((encoder_length , numLagFW)).tolist()  + lagX[i]
        y[i] = np.zeros((encoder_length, 1)).tolist() + y[i]
    return X, lagX, y

def mergeTrainValSets(XXTrain, lagXXTrain, YYTrain, XXValidation, lagXXValidation, YYValidation, encoder_length):
    for i in range(len(XXTrain)):
        XXTrain[i] = XXTrain[i] + XXValidation[i][encoder_length:]
        lagXXTrain[i] = lagXXTrain[i] + lagXXValidation[i][encoder_length:]
        YYTrain[i] = YYTrain[i] + YYValidation[i][encoder_length:]
    return XXTrain, lagXXTrain, YYTrain

def getValSet(XXTest, lagXXTest, YYTest, val_indices):
    XXValidation, lagXXValidation, YYValidation = list(), list(), list()
    XXTest_new, lagXXTest_new, YYTest_new = list(), list(), list()
    for i in range(len(XXTest)):
        if i in val_indices:
            XXValidation.append(XXTest[i])
            lagXXValidation.append(lagXXTest[i])
            YYValidation.append(YYTest[i])
        else:
            XXTest_new.append(XXTest[i])
            lagXXTest_new.append(lagXXTest[i])
            YYTest_new.append(YYTest[i])
    return XXValidation, lagXXValidation, YYValidation, XXTest_new, lagXXTest_new, YYTest_new

def getAdaptStatePlaceholders(params, input_dims):
    sxxt_p = tf.placeholder(tf.float32, [None,input_dims+1,input_dims+1])
    sxy_p = tf.placeholder(tf.float32, [None,input_dims+1,params.numY])
    count_p = tf.placeholder(tf.float32,[None])

    spxx_p = tf.placeholder(tf.float32, [None, max(1, params.numProjs)])
    spx_p = tf.placeholder(tf.float32, [None, max(1, params.numProjs)])
    spxy_p = tf.placeholder(tf.float32, [None, max(1, params.numProjs), params.numY])
    sy_p = tf.placeholder(tf.float32, [None,params.numY])

    return (sxxt_p,sxy_p,spx_p,spxx_p,sy_p,spxy_p,count_p)


def runModel(params,
             XXTrain, lagXXTrain, YYTrain,
             XXValidation, lagXXValidation, YYValidation,
             XXTest, lagXXTest, YYTest,
             maxYY, avgYY, val_indices, feature_dict, emb_dict):

        numFW = len(XXTrain[0][0]) if len(XXTrain[0]) else 0
        numLagFW = len(lagXXTrain[0][0])

        start_time = time.time()
        if params.valFrom == 'test':
            val_indices = sorted(np.random.randint(low=0, high=len(XXTrain)-1, size=int(len(XXTrain)/3)).tolist())
            XXValidation, lagXXValidation, YYValidation, XXTest, lagXXTest, YYTest = getValSet(XXTest, lagXXTest, YYTest, val_indices)
        # XXTrain, lagXXTrain, YYTrain = mergeTrainValSets(XXTrain, lagXXTrain, YYTrain, XXValidation, lagXXValidation, YYValidation, params.encoder_length)

        # Prepend zeros such that each y value gets to be part of the chunk.
        if params.isPrependZeros:
            XXTrain, lagXXTrain, YYTrain = prependZeros(XXTrain, lagXXTrain, YYTrain, params.encoder_length, numFW, numLagFW)


        inputPh, lagPh, outputPh, maxYYPh, keep_prob, mode, tsIndicesPh, maskPh \
                = getInputPlaceHolders(params.modelToRun, numFW, numLagFW, params.sequence_length, params.batch_size)

        if params.isAdapt:
            # First fix dimensions of ARU cell
            if params.modelToRun in ['aru', 'enc_aru', 'dec_aru', 'maruY', 'mEncAruY', 'localEncAru', 'localDecAru']:
                aruDims = numFW
            elif params.modelToRun in ['maruX_rnnAugment', 'mEncAruX_rnnAugment']:
                aruDims = numFW-1 + params.lstm_dense_layer_size
            elif params.modelToRun in ['maruX_sruAugment', 'mEncAruX_sruAugment']:
                aruDims = numFW-1 + len(alpha_list)
            elif params.modelToRun in ['mEncAruX', 'maruX']:
                aruDims = params.hidden2
            # adapt_tuple_placeholders = getAdaptStatePlaceholders(params, aruDims)
            elif params.modelToRun == 'adapt':
                #state_object = ar.AdaptiveRegression(params.hidden2)
                #adapt_tuple_placeholders = state_object.getAdaptStateTuplePlaceholders()
                aruDims =params.hidden2
        else:
            if params.modelToRun in ['neuralARU']:
                aruDims = numFW
            else:
                aruDims = 0


        model_obj = Models(params, inputPh, lagPh, outputPh, maxYYPh, tsIndicesPh, maskPh,
                           keep_prob, alpha_list, mode, numFW, aruDims, feature_dict, emb_dict, len(YYTrain))

        if params.modelToRun in ['enc_aru','dec_aru']:
            loss, pred, all_adapt_states = model_obj.getARUModel()
        elif params.modelToRun in ['maruY','maruX_sruAugment','maruX_rnnAugment','maruX']:
            loss, pred, all_adapt_states = model_obj.getModifiedARUModel()
        elif params.modelToRun in ['mEncAruY','mEncAruX_rnnAugment','mEncAruX_sruAugment','mEncAruX']:
            loss, pred, all_adapt_states = model_obj.getModifiedEncARUModel()
        elif params.modelToRun in ['localEncAru', 'localDecAru']:
            loss, pred, all_adapt_states = model_obj.getFullyLocalModel()
        elif params.modelToRun in ['neuralARU']:
            loss, pred = model_obj.getNeuralARUModel()
        elif params.modelToRun == 'adapt':
            loss, pred, all_adapt_states = model_obj.getAdaptModel()
        elif params.modelToRun == 'baseline':
            loss, pred = model_obj.getBaselineModel()
        elif params.modelToRun == 'hybrid':
            loss, pred = model_obj.getHybridModel()
        elif params.modelToRun == 'snail':
            loss, pred = model_obj.getSNAILModel()
        elif params.modelToRun == 'attention':
            loss, pred = model_obj.getAttentionHybridModel()

        if params.modelToRun not in ['localEncAru', 'localDecAru']:
            optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.66)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        session = tf.Session(config = config)
        session.run(tf.global_variables_initializer())

        # XXTest = prependHistoryToTest(XXTrain, XXTest, params.encoder_length)
        # YYTest = prependHistoryToTest(YYTrain, YYTest, params.encoder_length)

        def predict(sess, XX, lagXX, YY, mYY, aYY, stepLen=params.decoder_length):

            y_size = len(YY[0])
            for yy in YY:
                assert len(yy) == y_size

            YY_pred = [[] for _ in range(len(YY))]
            SS_pred = [[] for _ in range(len(YY))]
            xItr = BatchGenerator(XX, batchSeqLen=params.sequence_length)
            lagXItr = BatchGenerator(lagXX, batchSeqLen=params.sequence_length)
            outputItr = BatchGenerator(YY, batchSeqLen=params.sequence_length)

            currItr = xItr.iterFinished
            total_loss = 0
            while currItr == xItr.iterFinished:
                b_x1, _, _, _, _, _ = xItr.nextBatch(batchSize=params.batch_size, stepLen=stepLen)
                b_lagx1, _, _, _, _, _ = lagXItr.nextBatch(batchSize=params.batch_size, stepLen=stepLen)
                b_y1, tsIndices_, _, _, _, mask = outputItr.nextBatch(batchSize=params.batch_size, stepLen=stepLen)
                #b_mYY = [mYY[tsInd] for tsInd in tsIndices_]

                dictionary = {inputPh:b_x1, lagPh:b_lagx1, outputPh:b_y1, keep_prob:1.00, mode:0.0, tsIndicesPh:tsIndices_, maskPh:mask}

                if params.is_probabilistic_pred:
                    loss_ret, pred_ret, sigma_ret = session.run([loss, pred, model_obj.sigma],feed_dict=dictionary)
                else:
                    loss_ret, pred_ret = session.run([loss, pred], feed_dict=dictionary)
                    sigma_ret = -1.0 * np.ones_like(pred_ret) # invalid sigma
                total_loss += loss_ret

                y_pred, sigma_pred = pred_ret, sigma_ret
                for i, tsInd in enumerate(tsIndices_):
                    if mask[i] > 0:
                        YY_pred[tsInd] += y_pred[i].tolist()
                        SS_pred[tsInd] += sigma_pred[i].tolist()

            YY_gt = np.squeeze(np.array(YY)[:, params.encoder_length:, :], axis=-1)
            for yy, yy_pred in zip(YY_gt, YY_pred):
                assert len(yy) == len(yy_pred)
            YY_pred = np.array(YY_pred)
            SS_pred = np.array(SS_pred)
            assert YY_gt.shape[0] == YY_pred.shape[0]
            assert YY_gt.shape[1] == YY_pred.shape[1]

            if params.isNormalised:
                YY_gt = YY_gt * np.array(mYY)
                YY_pred = YY_pred * np.array(mYY)
            if params.deep_ar_normalize_seq:
                YY_gt = YY_gt * np.array(aYY)
                YY_pred = YY_pred * np.array(aYY)
            if params.isLogNormalised:
                YY_gt = np.exp(YY_gt)
                YY_pred = np.exp(YY_pred)

            error = np.square(YY_gt - YY_pred)
            test_len = YY_gt.shape[1]

            score = [metrics.RMSE(YY_gt, YY_pred),
                     metrics.RMSE(YY_gt[:, :1], YY_pred[:, :1]),
                     metrics.RMSE(YY_gt[:, :int(test_len/2.0)], YY_pred[:, :int(test_len/2.0)]),
                     metrics.RMSE(YY_gt[:, int(test_len/2.0):], YY_pred[:, int(test_len/2.0):])]

            print([metrics.ND(YY_gt, YY_pred),
                   metrics.ND(YY_gt[:, :1], YY_pred[:, :1]),
                   metrics.ND(YY_gt[:, :int(test_len/2.0)], YY_pred[:, :int(test_len/2.0)]),
                   metrics.ND(YY_gt[:, int(test_len/2.0):], YY_pred[:, int(test_len/2.0):])])

            return total_loss, score, YY_gt, YY_pred, SS_pred


        #xItr = BatchGenerator(XXTrain, batchSeqLen=params.sequence_length, stepLen=params.stepLen)
        #numBatches = 0
        #while xItr.iterFinished == 0:
        #    _, _, _ = xItr.nextBatch(batchSize=1)
        #    numBatches += 1
        #print('Number of Batches:', numBatches)
        testLosses, valLosses, testScores, valScores, YYTestGtList, YYTestPredList = list(), list(), list(), list(), list(), list()
        SSTestPredList = list()
        bch_shuffle_seed = 4
        xItr = BatchGenerator(XXTrain, batchSeqLen=params.sequence_length)#, shuffle_batches=True, seed=bch_shuffle_seed)
        lagXItr = BatchGenerator(lagXXTrain, batchSeqLen=params.sequence_length)#, shuffle_batches=True, seed=bch_shuffle_seed)
        outputItr = BatchGenerator(YYTrain, batchSeqLen=params.sequence_length)#, shuffle_batches=True, seed=bch_shuffle_seed)

        def updateStates(startingTs, endingTs, tsIndices_, newStates):
            for i in range(len(newStates)):
                outputItr.states[i][tsIndices_] = newStates[i]

        while xItr.iterFinished < params.num_iterations:
            it_loss = 0.0
            currItr = xItr.iterFinished
            batchId = 0
            epoch_st = time.time()
            st = time.time()
            while currItr==xItr.iterFinished:
                batchId += 1
                if batchId % 100 == 0 and params.verbose:
                    et = time.time()
                    print(str(batchId)+' batches completed, took '+str(et-st)+' seconds, Epoch Loss:'+str(it_loss))
                    st = time.time()

                batchX, _, _, _, _, _ = xItr.nextBatch(batchSize=params.batch_size, stepLen=params.stepLen)
                batchLagX, _, _, _, _, _ = lagXItr.nextBatch(batchSize=params.batch_size, stepLen=params.stepLen)
                batchOutput, tsIndices_, _, startingTs, endingTs, mask = outputItr.nextBatch(batchSize=params.batch_size, stepLen=params.stepLen)
                batchMaxYY = [maxYY[tsInd] for tsInd in tsIndices_]

                dictionary = {inputPh:batchX, lagPh:batchLagX, outputPh:batchOutput, maxYYPh:batchMaxYY,
                              keep_prob:0.6, mode:1.0, tsIndicesPh:tsIndices_, maskPh:mask}

                if params.isAdapt:
                    if params.modelToRun in ['localEncAru', 'localDecAru']:
                        y_pred_nn_ret, loss_ret, newStates = session.run([pred, loss, all_adapt_states], feed_dict=dictionary)
                    else:
                        if params.dataset in ['synthetic']:
                            newStates = session.run([all_adapt_states], feed_dict=dictionary)
                            _ = session.run([optimizer], feed_dict=dictionary)
                            y_pred_nn_ret, loss_ret = session.run([pred, loss], feed_dict=dictionary)
                        else:
                            _, y_pred_nn_ret, loss_ret, newStates = session.run([optimizer, pred, loss, all_adapt_states], feed_dict=dictionary)
                else:
                    _, y_pred_nn_ret, loss_ret = session.run([optimizer, pred, loss], feed_dict=dictionary)


                y_act = batchOutput[:, params.encoder_length:, :]
                y_act = np.squeeze(y_act, axis=-1)
                it_loss += loss_ret
                #if params.isAdapt:
                #    updateStates(startingTs, endingTs, tsIndices_, newStates)

            print('Iteration', currItr, 'Train Loss = ',it_loss)
            xItr.reset()
            lagXItr.reset()
            outputItr.reset()
            if params.valFrom == 'test':
                mYY_test = [mY for ind, mY in enumerate(maxYY) if ind not in val_indices]
                aYY_test = [aY for ind, aY in enumerate(avgYY) if ind not in val_indices]
                #mYY_val, aYY_val = maxYY, avgYY
            elif params.valFrom == 'train':
                mYY_test, aYY_test = maxYY, avgYY
            mYY_val = [mY for ind, mY in enumerate(maxYY) if ind in val_indices]
            aYY_val = [aY for ind, aY in enumerate(avgYY) if ind in val_indices]
            inference_st = time.time()
            test_loss, testScore, YYTestGt, YYTestPred, SSTestPred = predict(session, XXTest, lagXXTest, YYTest,
                                                                  mYY_test,
                                                                  aYY_test,
                                                                  stepLen=params.decoder_length)
            inference_et = time.time()
            print('Inference Time:', inference_et-inference_st)
            val_loss, valScore, YYValGt, YYValPred, SSValPred = predict(session, XXValidation, lagXXValidation, YYValidation,
                                                              mYY_val,
                                                              aYY_val,
                                                              stepLen=params.decoder_length)
            print('Test Loss:', testScore[0], testScore[1], testScore[2], testScore[3], test_loss)
            print('Val Loss:', valScore[0], valScore[1], valScore[2], valScore[3], val_loss)
            #print 'minalpha:', minalpha

            testLosses.append(test_loss)
            valLosses.append(val_loss)
            testScores.append(testScore)
            valScores.append(valScore)
            YYTestGtList.append(YYTestGt)
            YYTestPredList.append(YYTestPred)
            SSTestPredList.append(SSTestPred)
            epoch_et = time.time()
            if params.verbose:
                print('Epoch took '+str(epoch_et-epoch_st)+' seconds.')

            if params.isAdapt:
                session.run(model_obj.reset_op)

        #minValIndex, _ = min(enumerate([v[0] for v in valScores]), key=itemgetter(1))
        minValIndices, _ = zip(*sorted(enumerate(valLosses), key=itemgetter(1))[:5])
        minTestScore = np.mean(np.stack([testScores[minValIndex] for minValIndex in minValIndices], axis=0), axis=0).tolist()
        minValScore = np.mean(np.stack([valScores[minValIndex] for minValIndex in minValIndices], axis=0), axis=0).tolist()
        bestYYTestGt = np.mean(np.stack([YYTestGtList[minValIndex] for minValIndex in minValIndices], axis=2), axis=2)
        bestYYTestPred = np.mean(np.stack([YYTestPredList[minValIndex] for minValIndex in minValIndices], axis=2), axis=2)
        bestSSTestPred = np.mean(np.stack([SSTestPredList[minValIndex] for minValIndex in minValIndices], axis=2), axis=2)

        end_time = time.time()
        print(end_time - start_time) # Must print before minTestScore and minValScore
        #print([testScores[minValIndex] for minValIndex in minValIndices])
        #print(np.stack([testScores[minValIndex] for minValIndex in minValIndices], axis=0))
        #print([valScores[minValIndex] for minValIndex in minValIndices])
        #print(np.stack([valScores[minValIndex] for minValIndex in minValIndices], axis=0))
        print(minTestScore, minValScore)

        return bestYYTestGt, bestYYTestPred, bestSSTestPred
