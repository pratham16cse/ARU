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
import sru
import fru
from models import *
import pickle
from prepare_nn_features import CompetitionOpenSinceYear2int
from sklearn.preprocessing import StandardScaler
from rossman_data_utils import *

hidden1 = 10
hidden2 = 5
adam_l_rate = 0.0010
num_nodes = 20
num_layers = 1
lstm_dense_layer_size = 5
max_batch_size = 3
num_iterations = 10
batch_size = max_batch_size
no_of_iterations = num_iterations
sru_num_stats = 1500 #200 # 1000
alpha_list = [0, 0.25, 0.5, 0.9, 0.99]
recur_dims = 60
step_length = 1

def runAdaptStatelessLSTMRossman(encoder_length,decoder_length,sequence_length,count,isSRU,isSRUFeat):
	
	#RossmanSales
	dfTrain,dfTest,X_Test,salesLogMax,salesLogMin = getTrainTestData()
	features = getFeatures()

	tf.reset_default_graph()
	sequence_length = encoder_length + decoder_length
	
	max_no_stores = count
	adapt = True
	no_of_items = count
	embedding_size = 1


	input1 = tf.placeholder(tf.float32, (None, sequence_length, len(features)))
	output = tf.placeholder(tf.float32, (None, sequence_length))
	outputPrev = tf.placeholder(tf.float32, (None, sequence_length))

	keep_prob = tf.placeholder(tf.float32)
	structure = transformInput(input1)
	input11 = tf.concat(structure, axis=2)

	# state_placeholder = tf.placeholder(tf.float32, [num_layers, 2, None, num_nodes])
	sxxt_p = tf.placeholder(tf.float32, [None,None,None])
	sxy_p = tf.placeholder(tf.float32, [None,None])
	count_p = tf.placeholder(tf.float32,[None])

	Y_out_W_lstm = tf.Variable(tf.random_uniform([num_nodes, lstm_dense_layer_size]))
	Y_out_B_lstm = tf.Variable(tf.random_uniform([lstm_dense_layer_size]))


	inputS = input11


	mov_avg = tf.placeholder(tf.float32, [None,1])
	mult_scale_avg = tf.placeholder(tf.float32, [None,4])


	loss,loss2,pred,pred2,all_sxxt,all_sxy,all_count = getAdaptModel(input11,output,outputPrev,sxxt_p, sxy_p, count_p,mov_avg,mult_scale_avg,isSRU,isSRUFeat,num_layers,encoder_length,
					sequence_length,decoder_length,Y_out_W_lstm,Y_out_B_lstm,lstm_dense_layer_size,hidden1,hidden2,keep_prob,sru_num_stats,alpha_list,recur_dims,num_nodes)


	optimiser = tf.train.AdamOptimizer(learning_rate=adam_l_rate).minimize(loss)
	optimiser2 = tf.train.AdamOptimizer(learning_rate=adam_l_rate).minimize(loss2)

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
	config = tf.ConfigProto(gpu_options=gpu_options)
	config.gpu_options.allow_growth = True

	session = tf.Session(config=config)
	session.run(tf.global_variables_initializer())


	def startTrain(max_batch_size,sequence_length,session,iteration_):
	    dfTrain2 = dfTrain.sort_values(['store_index','year','month','day'])
	    
	    index_vals = []
	    l3 = []
	    l4 = []
	    l5 = []
	#     l6 = []
	    
	    for i in range(max_batch_size):
	        store_len = len(dfTrain2.loc[dfTrain2['store_index'] == i].values)-1
	#         index_vals.append((i,(store_len-decoder_length)%encoder_length))
	        index_vals.append((i,0))
	            
	    c_index = max_batch_size
	    total_loss = 0
	    while len(index_vals) > 0:
	        batch_x = []
	        batch_y = []
	        batch_mov_avg = []
	        batch_mult_scale_avg = []
	#         batch_xx = []
	        batch_y_prev = []
	        temp_index = []
	        
	        sxxt_needed = []
	        sxy_needed = []
	        count_needed = []
	        
	#         states_needed = []
	        
	        for k in range(len(index_vals)):
	            cur_ind = index_vals[k]
	            cur_ind_store = cur_ind[0]
	            cur_ind_start = cur_ind[1]
	            dfTrainItem = dfTrain2.loc[dfTrain2['store_index'] == cur_ind_store]
	            dfTrainItemXVals = dfTrainItem[features].values[1:]
	            dfTrainItemYVals = dfTrainItem['ScaledLogSales'].values[1:]
	            dfTrainItemPrevYVals = dfTrainItem['ScaledLogSales'].values[:-1]
	#             dfTrainItemXXVals = dfTrainItem[['store_index','day','month','ScaledLogSales']].values
	            item_length = len(dfTrainItemXVals)
	            if cur_ind_start + sequence_length >= item_length:
	                batch_x.append(dfTrainItemXVals[-sequence_length:])
	                batch_y.append(dfTrainItemYVals[-sequence_length:])
	                batch_y_prev.append(dfTrainItemPrevYVals[-sequence_length:])
	                yy_app = dfTrainItemYVals[-sequence_length:]
	                yy_app = np.reshape(yy_app,(yy_app.shape[0],1))
	                batch_mov_avg.append([moving_average(yy_app[0:encoder_length])])
	                batch_mult_scale_avg.append(multiScaleAverage(yy_app[0:encoder_length],alpha=[0.0,0.5,0.9,0.99]))

	#                 batch_xx.append(dfTrainItemXXVals[-sequence_length:])
	                
	#                 state_n = l6[-1]
	#                 state_n = getNumpyArrayFromState(state_n)
	#                 states_needed.append(state_n[:,:,k,:])
	                
	                if adapt:
	                    sxxt_n = l3[-1]
	                    sxxt_needed.append(sxxt_n[k,:,:])
	                    sxy_n = l4[-1]
	                    sxy_needed.append(sxy_n[k,:])
	                    count_n = l5[-1]
	                    count_needed.append(count_n[k])
	                
	                if c_index < max_no_stores:
	                    store_len = len(dfTrain2.loc[dfTrain2['store_index'] == c_index].values)
	#                     temp_index.append((c_index,(store_len-decoder_length)%encoder_length))
	                    temp_index.append((c_index,0))
	                    c_index+=1
	            elif cur_ind_start + sequence_length < item_length:
	                batch_x.append(dfTrainItemXVals[cur_ind_start:cur_ind_start+sequence_length])
	                batch_y.append(dfTrainItemYVals[cur_ind_start:cur_ind_start+sequence_length])
	                yy_app = dfTrainItemYVals[cur_ind_start:cur_ind_start+sequence_length]
	                yy_app = np.reshape(yy_app,(yy_app.shape[0],1))
	                batch_mov_avg.append([moving_average(yy_app[0:encoder_length])])
	                batch_mult_scale_avg.append(multiScaleAverage(yy_app[0:encoder_length],alpha=[0.0,0.5,0.9,0.99]))

	                batch_y_prev.append(dfTrainItemPrevYVals[cur_ind_start:cur_ind_start+sequence_length])
	#                 batch_xx.append(dfTrainItemXXVals[cur_ind_start:cur_ind_start+sequence_length])
	                temp_index.append((cur_ind_store,cur_ind_start+1))
	                
	#                 if cur_ind_start < sequence_length:
	#                     states_needed.append(np.zeros((num_layers, 2,num_nodes)))
	#                 else:
	#                     state_n = l6[-1]
	#                     state_n = getNumpyArrayFromState(state_n)
	#                     states_needed.append(state_n[:,:,k,:])
	                
	                if adapt:
	                    if cur_ind_start < sequence_length:
	                        sxxt_needed.append(np.zeros((hidden2+1, hidden2+1)))
	                        sxy_needed.append(np.zeros((hidden2+1)))
	                        count_needed.append(0)
	                    else:
	                        sxxt_n = l3[-1]
	                        sxxt_needed.append(sxxt_n[k,:,:])
	                        sxy_n = l4[-1]
	                        sxy_needed.append(sxy_n[k,:])
	                        count_n = l5[-1]
	                        count_needed.append(count_n[k])
	                        
	        batch_x = np.array(batch_x)
	        batch_y = np.array(batch_y)
	        batch_y_prev = np.array(batch_y_prev)
	#         batch_xx = np.array(batch_xx)
	        batch_y = np.reshape(batch_y,(batch_y.shape[0],batch_y.shape[1]))
	        batch_y_prev = np.reshape(batch_y_prev,(batch_y_prev.shape[0],batch_y_prev.shape[1]))
	        
	        batch_size_f = np.shape(batch_x)[0]
	        
	        if adapt:
	            sxxt_needed = np.array(sxxt_needed)
	            sxy_needed = np.array(sxy_needed)
	            count_needed = np.array(count_needed)
	        else:
	            sxxt_needed = np.zeros((batch_size_f,hidden2+1, hidden2+1))
	            sxy_needed = np.zeros((batch_size_f,hidden2+1))
	            count_needed = np.zeros((batch_size_f))
	        
	#         batch_states_f = np.zeros((num_layers, 2, batch_size_f, num_nodes))
	#         for ss in range(batch_size_f):
	#             batch_states_f[:,:,ss,:] = states_needed[ss]
	        
	        l1,l2,l3,l4,l5 = session.run([optimiser2,loss,all_sxxt,all_sxy,all_count], feed_dict={input1: batch_x,output: batch_y, keep_prob:1.00, 
	                                                                                            sxxt_p: sxxt_needed,sxy_p: sxy_needed,
	                                                                                            count_p: count_needed,outputPrev:batch_y_prev,mov_avg: batch_mov_avg,
																								mult_scale_avg: batch_mult_scale_avg})
	        
	        total_loss += l2
	        index_vals = temp_index
	    print "total_loss = ",total_loss


	# In[ ]:


	def startTest(max_batch_size,sequence_length,session,iteration_):
	    dfTrain2 = dfTrain.sort_values(['store_index','year','month','day'])
	    dfTest2 = dfTest.sort_values(['store_index','year','month','day'])
	    
	    index_vals = []
	    l3 = []
	    l4 = []
	    l5 = []
	#     l6 = []
	    
	    c_index = 0
	    ans_keys = {}
	    
	    while len(index_vals) < max_batch_size and c_index < max_no_stores:
	        dfS = dfTest2.loc[dfTest2['store_index'] == c_index]
	        if dfS.shape[0] != 0:
	            store_len = len(dfTrain2.loc[dfTrain2['store_index'] == c_index].values)-1+decoder_length
	#             index_vals.append((c_index,store_len%sequence_length))
	#             index_vals.append((c_index,(store_len-decoder_length)%encoder_length))
	            index_vals.append((c_index,0))
	        c_index += 1
	    
	    while len(index_vals) > 0:
	        batch_x = []
	        batch_y = []
	        batch_y_prev = []
	        temp_index = []
	        
	        sxxt_needed = []
	        sxy_needed = []
	        count_needed = []
	        batch_mov_avg = []
	        batch_mult_scale_avg = []
	        
	#         states_needed = []
	                
	        for k in range(len(index_vals)):
	            cur_ind = index_vals[k]
	            cur_ind_store = cur_ind[0]
	            cur_ind_start = cur_ind[1]
	            dfTrainItem = dfTrain2.loc[dfTrain2['store_index'] == cur_ind_store]
	            dfTrainItemXVals = dfTrainItem[features].values[1:]
	            dfTrainItemYVals = dfTrainItem['ScaledLogSales'].values[1:]
	            dfTrainItemPrevYVals = dfTrainItem['ScaledLogSales'].values[:-1]
	            dfTestItem = dfTest2.loc[dfTest2['store_index'] == cur_ind_store]
	            
	            cxpt = 0
	            for x_pt in dfTestItem[features].values:
	                dfTrainItemXVals = np.append(dfTrainItemXVals,[x_pt],0)
	                dfTrainItemYVals = np.append(dfTrainItemYVals,[0],0)
	                if cxpt == 0:
	                    dfTrainItemPrevYVals = np.append(dfTrainItemPrevYVals,[dfTrainItem['ScaledLogSales'].values[-1]],0)
	                else:
	                    dfTrainItemPrevYVals = np.append(dfTrainItemPrevYVals,[0],0)
	                cxpt+=1
	                    
	            item_length = len(dfTrainItemXVals)
	            
	            if cur_ind_start + sequence_length >= item_length:
	                batch_x.append(dfTrainItemXVals[-sequence_length:])
	                batch_y.append(dfTrainItemYVals[-sequence_length:])
	                batch_y_prev.append(dfTrainItemPrevYVals[-sequence_length:])
	                yy_app = dfTrainItemYVals[-sequence_length:]
	                yy_app = np.reshape(yy_app,(yy_app.shape[0],1))
	                batch_mov_avg.append([moving_average(yy_app[0:encoder_length])])
	                batch_mult_scale_avg.append(multiScaleAverage(yy_app[0:encoder_length],alpha=[0.0,0.5,0.9,0.99]))

	#                 state_n = l6[-1]
	#                 state_n = getNumpyArrayFromState(state_n)
	#                 states_needed.append(state_n[:,:,k,:])
	                
	                if adapt:
	                    sxxt_n = l3[-1]
	                    sxxt_needed.append(sxxt_n[k,:,:])
	                    sxy_n = l4[-1]
	                    sxy_needed.append(sxy_n[k,:])
	                    count_n = l5[-1]
	                    count_needed.append(count_n[k])
	                    
	                while len(temp_index) < max_batch_size and c_index < max_no_stores:
	                    dfS = dfTest2.loc[dfTest2['store_index'] == c_index]
	                    if dfS.shape[0] != 0:
	                        store_len = len(dfTrain2.loc[dfTrain2['store_index'] == c_index].values)-1+decoder_length
	#                         temp_index.append((c_index,(store_len-decoder_length)%encoder_length))
	                        temp_index.append((c_index,0))
	                        c_index += 1
	                        break
	                    else:
	                        c_index += 1
	            
	            elif cur_ind_start + sequence_length < item_length:
	                batch_x.append(dfTrainItemXVals[cur_ind_start:cur_ind_start+sequence_length])
	                batch_y.append(dfTrainItemYVals[cur_ind_start:cur_ind_start+sequence_length])
	                batch_y_prev.append(dfTrainItemPrevYVals[cur_ind_start:cur_ind_start+sequence_length])
	                yy_app = dfTrainItemYVals[cur_ind_start:cur_ind_start+sequence_length]
	                yy_app = np.reshape(yy_app,(yy_app.shape[0],1))
	                batch_mov_avg.append([moving_average(yy_app[0:encoder_length])])
	                batch_mult_scale_avg.append(multiScaleAverage(yy_app[0:encoder_length],alpha=[0.0,0.5,0.9,0.99]))

	#                 temp_index.append((cur_ind_store,cur_ind_start+encoder_length))
	                temp_index.append((cur_ind_store,cur_ind_start+1))
	                
	#                 if cur_ind_start < sequence_length:
	#                     states_needed.append(np.zeros((num_layers, 2,num_nodes)))
	#                 else:
	#                     state_n = l6[-1]
	#                     state_n = getNumpyArrayFromState(state_n)
	#                     states_needed.append(state_n[:,:,k,:])
	                
	                if adapt:
	                    if cur_ind_start < sequence_length:
	                        sxxt_needed.append(np.zeros((hidden2+1, hidden2+1)))
	                        sxy_needed.append(np.zeros((hidden2+1)))
	                        count_needed.append(0)
	                    else:
	                        sxxt_n = l3[-1]
	                        sxxt_needed.append(sxxt_n[k,:,:])
	                        sxy_n = l4[-1]
	                        sxy_needed.append(sxy_n[k,:])
	                        count_n = l5[-1]
	                        count_needed.append(count_n[k])    
	        
	        
	        batch_x = np.array(batch_x)
	        batch_y = np.array(batch_y)
	        batch_y_prev = np.array(batch_y_prev)
	        batch_y = np.reshape(batch_y,(batch_y.shape[0],batch_y.shape[1]))
	        batch_y_prev = np.reshape(batch_y_prev,(batch_y_prev.shape[0],batch_y_prev.shape[1]))
	        
	        batch_size_f = np.shape(batch_x)[0]
	        
	        if adapt:
	            sxxt_needed = np.array(sxxt_needed)
	            sxy_needed = np.array(sxy_needed)
	            count_needed = np.array(count_needed)
	        else:
	            sxxt_needed = np.zeros((batch_size_f,hidden2+1, hidden2+1))
	            sxy_needed = np.zeros((batch_size_f,hidden2+1))
	            count_needed = np.zeros((batch_size_f))
	        
	        l1,l2,l3,l4,l5 = session.run([pred,loss,all_sxxt,all_sxy,all_count], feed_dict={input1: batch_x,output: batch_y, keep_prob:1.00, 
	                                                                                            sxxt_p: sxxt_needed,sxy_p: sxy_needed,
	                                                                                            count_p: count_needed,outputPrev:batch_y_prev,mov_avg: batch_mov_avg,
																								mult_scale_avg: batch_mult_scale_avg})
	        if len(np.shape(l1)) == 1:
	            for k in range(len(index_vals)):
	                cur_ind = index_vals[k]
	                cur_ind_store = cur_ind[0]
	                cur_ind_start = cur_ind[1]
	                dfTrainItem = dfTrain2.loc[dfTrain2['store_index'] == cur_ind_store]
	                dfTestItem = dfTest2.loc[dfTest2['store_index'] == cur_ind_store]
	                item_length = len(dfTrainItem.values)+len(dfTestItem.values)-1

	                if cur_ind_start + sequence_length >= item_length:
	                    indices = dfTestItem.index.values
	                    yy_pred = l1
	                    for kk in range(sequence_length):
	                        yy_pred_i = yy_pred[kk]*(salesLogMax)
	                        y_ = np.exp(yy_pred_i)
	                        ans_keys[indices[kk]] = y_
	        else:
	            for k in range(len(index_vals)):
	                cur_ind = index_vals[k]
	                cur_ind_store = cur_ind[0]
	                cur_ind_start = cur_ind[1]
	                dfTrainItem = dfTrain2.loc[dfTrain2['store_index'] == cur_ind_store]
	                dfTestItem = dfTest2.loc[dfTest2['store_index'] == cur_ind_store]
	                item_length = len(dfTrainItem.values)+len(dfTestItem.values)-1

	                if cur_ind_start + sequence_length >= item_length:
	                    indices = dfTestItem.index.values
	                    yy_pred = l1[k]
	                    for kk in range(decoder_length):
	                        yy_pred_i = yy_pred[kk]*(salesLogMax)
	                        y_ = np.exp(yy_pred_i)
	                        ans_keys[indices[kk]] = y_
	                        
	        index_vals = temp_index
	        
	    with open("predictions"+str(iteration_)+".csv", 'w') as f:
	        f.write('Id,Sales\n')
	        for i,record in enumerate(X_Test):
	            if ans_keys.has_key(i):
	                y_ = ans_keys[i]
	                store_num = record[1]-1
	                is_open = record[0]
	                if is_open == 0:
	                    y_ = 0                    
	                f.write('{},{}\n'.format(i+1, y_))
	    print "Finished test values"


	# In[ ]:


	for nn in range(num_iterations):
	    print "@@@@@@@@@@@"
	    startTrain(max_batch_size,sequence_length,session,nn)
	    startTest(max_batch_size,sequence_length,session,nn)



def runRossmanBaselineModel(encoder_length,decoder_length,sequence_length,count,isSRU,isSRUFeat):
	max_store_num = count
	maxS = 100
	df,dfTest,X_Test,salesLogMax,salesLogMin = getTrainTestData()
	features = getFeatures()

	tf.reset_default_graph()
	sequence_length = encoder_length + decoder_length
	
	max_no_stores = count
	adapt = True
	no_of_items = count


	input1 = tf.placeholder(tf.float32, (None,sequence_length,len(features)))
	input2 = tf.placeholder(tf.float32, (None,sequence_length,1))
	output = tf.placeholder(tf.float32, (None,sequence_length,1))

	structure = transformInput(input1)
	concat_input = tf.concat(structure, axis=2)



	Y_out_W_lstm = tf.Variable(tf.random_uniform([num_nodes,lstm_dense_layer_size]))
	Y_out_B_lstm = tf.Variable(tf.random_uniform([lstm_dense_layer_size]))
	Y_out_W_nn = tf.Variable(tf.random_uniform([hidden2,1]))
	Y_out_B_nn = tf.Variable(tf.random_uniform([1]))


	mov_avg = tf.placeholder(tf.float32, [None,1])
	mult_scale_avg = tf.placeholder(tf.float32, [None,4])


	pred,lossC = getBaselineModel(concat_input,input2,output,num_layers,Y_out_W_lstm,Y_out_B_lstm,lstm_dense_layer_size,isSRU,isSRUFeat,mov_avg,
		mult_scale_avg,hidden1,hidden2,Y_out_W_nn,Y_out_B_nn,sru_num_stats,alpha_list,recur_dims,num_nodes,sequence_length)

	
	optimiser = tf.train.AdamOptimizer(learning_rate=adam_l_rate).minimize(lossC)


	# In[ ]:


	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
	config = tf.ConfigProto(gpu_options=gpu_options)
	config.gpu_options.allow_growth = True
	session = tf.Session(config = config)
	session.run(tf.global_variables_initializer())


	def getTrainDataForIndex(dataF,i_index):
	    dfS = dataF.loc[dataF['store_index'] == i_index]
	    return dfS


	# In[ ]:


	def predictTestValues(sess,name):
	    Y_Test = []
	    ans_keys = {}
	    for i in range(count):
	        dfS = dfTest.loc[dfTest['store_index'] == i]
	        if dfS.shape[0] != 0:
	            dfT = df.loc[df['store_index'] == i]
	            y_next = dfT['ScaledLogSales'].values[-1*(sequence_length):]
	            y_prev = dfT['ScaledLogSales'].values[-1*(sequence_length):]
	            x_vals = dfT[features].values[-1*(sequence_length)+1:]
	            Y_out = []
	            indices = dfS.index.values
	            for ind in range(dfS.values.shape[0]):
	                x_pt = dfS.values[ind]
	                index = indices[ind]
	                if x_pt[0] != 0:
	                    x_vals = np.append(x_vals,[x_pt],0)
	                    b_x1 = []
	                    b_x2 = []
	                    b_y1 = []
	                    batch_mov_avg = []
	                    batch_mult_scale_avg = []
	                    b_x1.append(x_vals)
	                    b_x2.append(y_prev)
	                    b_y1.append(y_next)
	                    yy_app = y_prev
	                    yy_app = np.reshape(yy_app,(yy_app.shape[0],1))
	                    batch_mov_avg.append([moving_average(yy_app[0:encoder_length])])
	                    batch_mult_scale_avg.append(multiScaleAverage(yy_app[0:encoder_length],alpha=[0.0,0.5,0.9,0.99]))
	                    b_x1 = np.array(b_x1)
	                    b_x2 = np.array(b_x2)
	                    b_y1 = np.array(b_y1)
	                    b_x2 = b_x2.reshape(b_x2.shape[0],b_x2.shape[1],1)
	                    b_y1 = b_y1.reshape(b_y1.shape[0],b_y1.shape[1],1)
	                    Y_out = session.run([pred],feed_dict={input1: b_x1,input2: b_x2,output: b_y1,mov_avg: batch_mov_avg,mult_scale_avg: batch_mult_scale_avg})
	                    z_ = Y_out[0][0][0]
	#                     print "@@@@@@@z_ = ",z_
	                    z = z_*(salesLogMax - salesLogMin) + salesLogMin
	                    z = np.exp(z)
	                    ans_keys[index] = z
	                    x_vals = np.delete(x_vals,[0],0)
	                    y_prev = np.delete(y_prev,[0],0)
	                    y_prev = np.append(y_prev,[Y_out[0][0][0]],0)
	                else:
	                    ans_keys[index] = 0
	    with open(name, 'w') as f:
	        f.write('Id,Sales\n')
	        for i,record in enumerate(X_Test):
	            y_ = ans_keys[i]
	            f.write('{},{}\n'.format(i+1, y_))


	# In[ ]:


	n_it = 0
	while n_it < no_of_iterations:
	    n_it = n_it+1
	    it_loss = 0
	    en = sequence_length
	    print "iteration = ",n_it
	#     predictTestValues(session,"predictions_exp_lstm_baseline"+str(n_it)+".csv")
	    print "starting training"
	    while en < maxS:
	        i_index = 0
	        t_loss = 0
	        while i_index < max_store_num:
	            print "!",i_index,t_loss
	            bs = 0
	            batch_x = []
	            batch_x2 = []
	            batch_y = []
	            batch_mov_avg = []
	            batch_mult_scale_avg = []
	            while bs < batch_size:
	                if i_index >= max_store_num:
	                    break
	                else:
	                    dfS = getTrainDataForIndex(df,i_index)
	                    y_next = dfS['ScaledLogSales'].values[1:]
	                    y_prev = dfS['ScaledLogSales'].values[:-1]
	                    x_vals = dfS[features].values[1:]
	                    i_index = i_index+1
	                    if x_vals.shape[0] >= en:
	                        bs = bs+1
	                        input_x1 = x_vals[en-sequence_length:en]
	                        input_y2 = y_prev[en-sequence_length:en]
	                        output_y = y_next[en-sequence_length:en]
	                        batch_x.append(input_x1)
	                        batch_x2.append(input_y2)
	                        batch_y.append(output_y)
	                        yy_app = y_prev
	                        yy_app = np.reshape(yy_app,(yy_app.shape[0],1))
	                        batch_mov_avg.append([moving_average(yy_app[0:encoder_length])])
	                        batch_mult_scale_avg.append(multiScaleAverage(yy_app[0:encoder_length],alpha=[0.0,0.5,0.9,0.99]))
	            batch_x = np.array(batch_x)
	            batch_x2 = np.array(batch_x2)
	            batch_y = np.array(batch_y)
	            if batch_x.shape[0] > 0:
	                batch_x2 = batch_x2.reshape(batch_x2.shape[0],batch_x2.shape[1],1)
	                batch_y = batch_y.reshape(batch_y.shape[0],batch_y.shape[1],1)
	                l1,l2 = session.run([optimiser,lossC], feed_dict={input1: batch_x,input2: batch_x2,output: batch_y,mov_avg: batch_mov_avg,mult_scale_avg: batch_mult_scale_avg})
	                t_loss = t_loss+l2        
	        print "en = ",en,"it =",n_it
	        en = en+step_length
	    predictTestValues(session,"predictions_exp6"+str(n_it)+".csv")




def runRossmanHierarchicalModel(encoder_length,decoder_length,sequence_length,count,isSRU,isSRUFeat):
	
	n_test_len = decoder_length
	maxS = 100
	
	max_store_num = count
	no_of_items = count
	dfTrain,dfTest,X_Test,salesLogMax,salesLogMin = getTrainTestData()
	features = getFeatures()

	input1 = tf.placeholder(tf.float32, (None,None,len(features)))
	input2 = tf.placeholder(tf.float32, (None,None))
	output = tf.placeholder(tf.float32, (None,None,1))

	structure = transformInput(input1)
	concat_input = tf.concat(structure, axis=2)

	mov_avg = tf.placeholder(tf.float32, [None,1])
	mult_scale_avg = tf.placeholder(tf.float32, [None,4])


	loss,pred = getHierarchicalModel(concat_input, input2, output, sequence_length, encoder_length,decoder_length, hidden1, hidden2, num_nodes, num_layers, lstm_dense_layer_size,
		mov_avg,mult_scale_avg,sru_num_stats,alpha_list,recur_dims,isSRU,isSRUFeat)

	# In[ ]:

	lossC = tf.reduce_sum(loss)
	optimiser = tf.train.AdamOptimizer(learning_rate=adam_l_rate).minimize(lossC)

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
	config = tf.ConfigProto(gpu_options=gpu_options)
	config.gpu_options.allow_growth = True

	session = tf.Session(config=config)
	session.run(tf.global_variables_initializer())


		# In[ ]:
	def getTrainDataForIndex(dataF,i_index):
	    dfS = dataF.loc[dataF['store_index'] == i_index]
	    return dfS


	def predictTestValues(sess,name):
	    ans_keys = {}
	    i_index = 0
	    while i_index < max_store_num:
	        bs = 0
	        batch_x = []
	        batch_x2 = []
	        batch_y = []
	        batch_index = []
	        batch_mov_avg = []
	        batch_mult_scale_avg = []
	        while bs < batch_size:
	            if i_index >= max_store_num:
	                break
	            dfS = dfTest.loc[dfTest['store_index'] == i_index]
	            if dfS.shape[0] != 0:
	                dfT = dfTrain.loc[dfTrain['store_index'] == i_index]
	                y_next = dfT['ScaledLogSales'].values[-1*(sequence_length):]
	                y_prev = dfT['ScaledLogSales'].values[-1*(encoder_length)-1:]
	                x_vals = dfT[features].values[-1*(encoder_length):]
	                for x_pt in dfS.values:
	                    x_vals = np.append(x_vals,[x_pt],0)
	                for i in range(sequence_length-encoder_length-1):
	                    y_prev = np.append(y_prev,[0],0)
	                i_index = i_index + 1
	                bs = bs+1
	                yy_app = y_prev
	                yy_app = np.reshape(yy_app,(yy_app.shape[0],1))
	                batch_mov_avg.append([moving_average(yy_app[0:encoder_length])])
	                batch_mult_scale_avg.append(multiScaleAverage(yy_app[0:encoder_length],alpha=[0.0,0.5,0.9,0.99]))
	                batch_x.append(x_vals)
	                batch_x2.append(y_prev)
	                batch_y.append(y_next)
	                indices = dfS.index.values
	                batch_index.append(indices)
	            else:
	                i_index = i_index + 1
	        batch_x = np.array(batch_x)
	        batch_x2 = np.array(batch_x2)
	        batch_y = np.array(batch_y)
	        if batch_x.shape[0] > 0:
	            batch_x2 = batch_x2.reshape(batch_x2.shape[0], batch_x2.shape[1])
	            batch_y = batch_y.reshape(batch_y.shape[0], batch_y.shape[1],1)
	            y_list_test = sess.run([pred], feed_dict={input1: batch_x, input2: batch_x2, output: batch_y,mov_avg: batch_mov_avg,mult_scale_avg: batch_mult_scale_avg})
	            for kkk in range(np.shape(batch_index)[0]):
	                for rrr in range(np.shape(batch_index)[1]):
	                    bbb = batch_index[kkk][rrr]
	                    yyy = y_list_test[0][rrr][kkk][0]
	                    ans_keys[bbb] = yyy
	                    
	    with open(name, 'w') as f:
	        f.write('Id,Sales\n')
	        for i,record in enumerate(X_Test):
	            y_ = ans_keys[i]
	            store_num = record[1]-1
	            is_open = record[0]
	            if is_open != 0:
	                dfT = dfTrain.loc[dfTrain['store_index'] == store_num]
	#                 y_ = y_*(salesLogMax - salesLogMin) + salesLogMin
	                y_ = y_*(salesLogMax)
	                y_ = np.exp(y_)
	            else:
	                y_ = 0
	            f.write('{},{}\n'.format(i+1, y_))
	    return



	n_it = 0
	while n_it < no_of_iterations:
	    n_it = n_it+1
	    it_loss = 0
	    en = sequence_length
	    print "iteration = ",n_it
	#     predictTestValues(session,"predictions_exp6"+str(n_it)+".csv")
	    while en < maxS:
	        i_index = 0
	        t_loss = 0
	        while i_index < max_store_num:
	            print "!",i_index,t_loss
	            bs = 0
	            batch_x = []
	            batch_x2 = []
	            batch_y = []
	            batch_mov_avg = []
	            batch_mult_scale_avg = []
	            while bs < batch_size:
	                if i_index >= max_store_num:
	                    break
	                else:
	                    dfS = getTrainDataForIndex(dfTrain,i_index)
	                    y_next = dfS['ScaledLogSales'].values[1:]
	                    y_prev = dfS['ScaledLogSales'].values[:-1]
	                    x_vals = dfS[features].values[1:]
	                    i_index = i_index+1
	                    if x_vals.shape[0] >= en:
	                        bs = bs+1
	                        input_x1 = x_vals[en-sequence_length:en]
	                        input_y2 = y_prev[en-sequence_length:en]
	                        output_y = y_next[en-sequence_length:en]
	                        yy_app = output_y
	                        yy_app = np.reshape(yy_app,(yy_app.shape[0],1))
	                        batch_mov_avg.append([moving_average(yy_app[0:encoder_length])])
	                        batch_mult_scale_avg.append(multiScaleAverage(yy_app[0:encoder_length],alpha=[0.0,0.5,0.9,0.99]))
	                        batch_x.append(input_x1)
	                        batch_x2.append(input_y2)
	                        batch_y.append(output_y)
	            batch_x = np.array(batch_x)
	            batch_x2 = np.array(batch_x2)
	            batch_y = np.array(batch_y)
	            if batch_x.shape[0] > 0:
	                batch_x2 = batch_x2.reshape(batch_x2.shape[0],batch_x2.shape[1])
	                batch_y = batch_y.reshape(batch_y.shape[0],batch_y.shape[1],1)
	                l1,l2 = session.run([optimiser,lossC], feed_dict={input1: batch_x,input2: batch_x2,output: batch_y,mov_avg: batch_mov_avg,mult_scale_avg: batch_mult_scale_avg})
	                t_loss = t_loss+l2
	        print "en = ",en,"it =",n_it
	        en = en+step_length
	    predictTestValues(session,"predictions_exp6"+str(n_it)+".csv")

def runRossmanHybridModel(encoder_length,decoder_length,sequence_length,count,isSRU,isSRUFeat):

	n_test_len = decoder_length
	maxS = 100
	
	max_store_num = count
	no_of_items = count
	dfTrain,dfTest,X_Test,salesLogMax,salesLogMin = getTrainTestData()
	df = dfTrain
	features = getFeatures()

	input1 = tf.placeholder(tf.float32, (None,None,len(features)))
	input2 = tf.placeholder(tf.float32, (None,None))
	output = tf.placeholder(tf.float32, (None,None,1))

	structure = transformInput(input1)
	concat_input = tf.concat(structure, axis=2)

	Y_out_W_lstm = tf.Variable(tf.random_uniform([num_nodes, lstm_dense_layer_size]))
	Y_out_B_lstm = tf.Variable(tf.random_uniform([lstm_dense_layer_size]))
	Y_out_W_nn = tf.Variable(tf.random_uniform([hidden2,1]))
	Y_out_B_nn = tf.Variable(tf.random_uniform([1]))


	mov_avg = tf.placeholder(tf.float32, [None,1])
	mult_scale_avg = tf.placeholder(tf.float32, [None,4])


	pred,totalLoss = getHybridModel(concat_input,input2,output,mov_avg,mult_scale_avg,sequence_length,encoder_length,sru_num_stats,
		alpha_list,recur_dims,num_nodes,num_layers,hidden1,hidden2,isSRU,isSRUFeat,Y_out_W_lstm,Y_out_B_lstm,Y_out_W_nn,Y_out_B_nn,lstm_dense_layer_size)

	# In[ ]:


	optimiser = tf.train.AdamOptimizer(learning_rate=adam_l_rate).minimize(totalLoss)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
	config = tf.ConfigProto(gpu_options=gpu_options)
	config.gpu_options.allow_growth = True

	session = tf.Session(config = config)
	session.run(tf.global_variables_initializer())


	# In[ ]:
	def getTrainDataForIndex(dataF,i_index):
	    dfS = dataF.loc[dataF['store_index'] == i_index]
	    return dfS

	def predictTestValues(sess,name):
	    Y_Test = []
	    ans_keys = {}
	    Y_ans = np.empty((0, 2))
	    for i in range(max_store_num):
	        dfS = dfTest.loc[dfTest['store_index'] == i]
	        if dfS.shape[0] != 0:
	            dfT = df.loc[df['store_index'] == i]
	            y_next = dfT['ScaledLogSales'].values[-1*(sequence_length):]
	            y_prev = dfT['ScaledLogSales'].values[-1*(encoder_length)-1:]
	            x_vals = dfT[features].values[-1*(encoder_length):]
	            for x_pt in dfS.values:
	                x_vals = np.append(x_vals,[x_pt],0)
	            for i in range(sequence_length-encoder_length-1):
	                y_prev = np.append(y_prev,[0],0)
	            b_x1 = []
	            b_x2 = []
	            b_y1 = []
	            batch_mov_avg = []
	            batch_mult_scale_avg = []
	            b_x1.append(x_vals)
	            b_x2.append(y_prev)
	            b_y1.append(y_next)
	            yy_app = y_prev
	            yy_app = np.reshape(yy_app,(yy_app.shape[0],1))
	            batch_mov_avg.append([moving_average(yy_app[0:encoder_length])])
	            batch_mult_scale_avg.append(multiScaleAverage(yy_app[0:encoder_length],alpha=[0.0,0.5,0.9,0.99]))
	            b_x1 = np.array(b_x1)
	            b_x2 = np.array(b_x2)
	            b_y1 = np.array(b_y1)
	            b_x2 = b_x2.reshape(b_x2.shape[0],b_x2.shape[1])
	            b_y1 = b_y1.reshape(b_y1.shape[0],b_y1.shape[1],1)
	            predictions = session.run([pred],feed_dict={input1: b_x1,input2: b_x2,output: b_y1,mov_avg: batch_mov_avg,mult_scale_avg: batch_mult_scale_avg})
	            indices = dfS.index.values
	            for indexC in range(indices.shape[0]):
	                ans_keys[indices[indexC]] = predictions[0][indexC][0][0]

	    with open(name, 'w') as f:
	        f.write('Id,Sales\n')
	        for i,record in enumerate(X_Test):
	            y_ = ans_keys[i]
	            store_num = record[1]-1
	            is_open = record[0]
	            if is_open != 0:
	                dfT = df.loc[df['store_index'] == store_num]
	                y_ = y_*(salesLogMax - salesLogMin) + salesLogMin
	                y_ = np.exp(y_)
	            else:
	                y_ = 0
	            f.write('{},{}\n'.format(i+1, y_))
	    return


	# In[ ]:


	n_it = 0
	while n_it < no_of_iterations:
	    n_it = n_it+1
	    it_loss = 0
	    en = sequence_length
	    print "iteration = ",n_it
	    while en < maxS:
	        i_index = 0
	        t_loss = 0
	        while i_index < max_store_num:
	            print "!",i_index,t_loss
	            bs = 0
	            batch_x = []
	            batch_x2 = []
	            batch_y = []
	            batch_mov_avg = []
	            batch_mult_scale_avg = []
	            while bs < batch_size:
	                if i_index >= max_store_num:
	                    break
	                else:
	                    dfS = getTrainDataForIndex(df,i_index)
	                    y_next = dfS['ScaledLogSales'].values[1:]
	                    y_prev = dfS['ScaledLogSales'].values[:-1]
	                    x_vals = dfS[features].values[1:]
	                    i_index = i_index+1
	                    if x_vals.shape[0] >= en:
	                        bs = bs+1
	                        input_x1 = x_vals[en-sequence_length:en]
	                        input_y2 = y_prev[en-sequence_length:en]
	                        output_y = y_next[en-sequence_length:en]
	                        batch_x.append(input_x1)
	                        batch_x2.append(input_y2)
	                        batch_y.append(output_y)
	                        yy_app = y_prev
	                        yy_app = np.reshape(yy_app,(yy_app.shape[0],1))
	                        batch_mov_avg.append([moving_average(yy_app[0:encoder_length])])
	                        batch_mult_scale_avg.append(multiScaleAverage(yy_app[0:encoder_length],alpha=[0.0,0.5,0.9,0.99]))
	            batch_x = np.array(batch_x)
	            batch_x2 = np.array(batch_x2)
	            batch_y = np.array(batch_y)
	            if batch_x.shape[0] > 0:
	                batch_x2 = batch_x2.reshape(batch_x2.shape[0],batch_x2.shape[1])
	                batch_y = batch_y.reshape(batch_y.shape[0],batch_y.shape[1],1)
	                l1,l2 = session.run([optimiser,totalLoss], feed_dict={input1: batch_x,input2: batch_x2,output: batch_y,mov_avg: batch_mov_avg,mult_scale_avg: batch_mult_scale_avg})
	                t_loss = t_loss+l2
	        print "en = ",en,"it =",n_it
	        en = en+step_length
	    predictTestValues(session,"predictions_exp6"+str(n_it)+".csv")

	

def runRossmanAttentionModel(encoder_length,decoder_length,sequence_length,count,isSRU,isSRUFeat):

	dfTrain,dfTest,X_Test,salesLogMax,salesLogMin = getTrainTestData()
	features = getFeatures()

	
	ATTENTION_SIZE = 5
	maxS = 100
	max_no_stores = count
	no_of_items = count
	n_test_len = decoder_length

	input1 = tf.placeholder(tf.float32, (None, sequence_length, len(features)))
	output = tf.placeholder(tf.float32, (None, sequence_length))
	outputPrev = tf.placeholder(tf.float32, (None, sequence_length))

	structure = transformInput(input1)
	input11 = tf.concat(structure, axis=2)


	Y_out_W_lstm = tf.Variable(tf.random_uniform([num_nodes, lstm_dense_layer_size]))
	Y_out_B_lstm = tf.Variable(tf.random_uniform([lstm_dense_layer_size]))
	Y_out_W_nn = tf.Variable(tf.random_uniform([hidden2,1]))
	Y_out_B_nn = tf.Variable(tf.random_uniform([1]))


	predictions,totalLoss = getAttentionHybridModel(input11,outputPrev,output,encoder_length,sequence_length,sru_num_stats,alpha_list,recur_dims,num_nodes,num_layers,
		ATTENTION_SIZE,Y_out_W_lstm,Y_out_B_lstm,lstm_dense_layer_size,hidden1,hidden2,Y_out_W_nn,Y_out_B_nn)

	# In[ ]:


	optimiser = tf.train.AdamOptimizer(learning_rate=adam_l_rate).minimize(totalLoss)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
	config = tf.ConfigProto(gpu_options=gpu_options)
	config.gpu_options.allow_growth = True

	session = tf.Session(config = config)
	session.run(tf.global_variables_initializer())


	# In[ ]:

	def getTrainDataForIndex(dataF,i_index):
	    dfS = dataF.loc[dataF['store_index'] == i_index]
	    return dfS

	def predictTestValues(sess,name):
	    Y_Test = []
	    ans_keys = {}
	    Y_ans = np.empty((0, 2))
	    for i in range(max_no_stores):
	        dfS = dfTest.loc[dfTest['store_index'] == i]
	        if dfS.shape[0] != 0:
	            dfT = dfTrain.loc[dfTrain['store_index'] == i]
	            y_next = dfT['ScaledLogSales'].values[-1*(sequence_length):]
	            y_prev = dfT['ScaledLogSales'].values[-1*(encoder_length)-1:]
	            x_vals = dfT[features].values[-1*(encoder_length):]
	            for x_pt in dfS.values:
	                x_vals = np.append(x_vals,[x_pt],0)
	            for i in range(sequence_length-encoder_length-1):
	                y_prev = np.append(y_prev,[0],0)
	            b_x1 = []
	            b_x2 = []
	            b_y1 = []
	            batch_mov_avg = []
	            batch_mult_scale_avg = []
	            b_x1.append(x_vals)
	            b_x2.append(y_prev)
	            b_y1.append(y_next)
	            yy_app = y_prev
	            yy_app = np.reshape(yy_app,(yy_app.shape[0],1))
	            batch_mov_avg.append([moving_average(yy_app[0:encoder_length])])
	            batch_mult_scale_avg.append(multiScaleAverage(yy_app[0:encoder_length],alpha=[0.0,0.5,0.9,0.99]))
	            b_x1 = np.array(b_x1)
	            b_x2 = np.array(b_x2)
	            b_y1 = np.array(b_y1)
	            b_x2 = b_x2.reshape(b_x2.shape[0],b_x2.shape[1])
	            b_y1 = b_y1.reshape(b_y1.shape[0],b_y1.shape[1])
	            predictionsT,total_loss = session.run([predictions,totalLoss],feed_dict={input1: b_x1,outputPrev: b_x2,output: b_y1})
	            indices = dfS.index.values
	            print np.shape(predictionsT)
	            for indexC in range(indices.shape[0]):
	                ans_keys[indices[indexC]] = predictionsT[indexC][0][0]
	    with open(name, 'w') as f:
	        f.write('Id,Sales\n')
	        for i,record in enumerate(X_Test):
	            y_ = ans_keys[i]
	            print "wtf",y_
	            store_num = record[1]-1
	            is_open = record[0]
	            if is_open != 0:
	                dfT = dfTrain.loc[dfTrain['store_index'] == store_num]
	                y_ = y_*(salesLogMax - salesLogMin) + salesLogMin
	                y_ = np.exp(y_)
	            else:
	                y_ = 0
	            f.write('{},{}\n'.format(i+1, y_))
	    return


	# In[ ]:


	n_it = 0
	while n_it < num_iterations:
	    n_it = n_it+1
	    it_loss = 0
	    en = sequence_length
	    print "iteration = ",n_it
	    while en < maxS:
	        i_index = 0
	        t_loss = 0
	        while i_index < max_no_stores:
	            print "!",i_index,t_loss
	            bs = 0
	            batch_x = []
	            batch_x2 = []
	            batch_y = []
	            batch_mov_avg = []
	            batch_mult_scale_avg = []
	            while bs < max_batch_size:
	                if i_index >= max_no_stores:
	                    break
	                else:
	                    dfS = getTrainDataForIndex(dfTrain,i_index)
	                    y_next = dfS['ScaledLogSales'].values[1:]
	                    y_prev = dfS['ScaledLogSales'].values[:-1]
	                    x_vals = dfS[features].values[1:]
	                    i_index = i_index+1
	                    if x_vals.shape[0] >= en:
	                        bs = bs+1
	                        input_x1 = x_vals[en-sequence_length:en]
	                        input_y2 = y_prev[en-sequence_length:en]
	                        output_y = y_next[en-sequence_length:en]
	                        batch_x.append(input_x1)
	                        batch_x2.append(input_y2)
	                        batch_y.append(output_y)
	                        yy_app = output_y
	                        yy_app = np.reshape(yy_app,(yy_app.shape[0],1))
	                        batch_mov_avg.append([moving_average(yy_app[0:encoder_length])])
	                        batch_mult_scale_avg.append(multiScaleAverage(yy_app[0:encoder_length],alpha=[0.0,0.5,0.9,0.99]))
	            batch_x = np.array(batch_x)
	            batch_x2 = np.array(batch_x2)
	            batch_y = np.array(batch_y)
	            if batch_x.shape[0] > 0:
	                batch_x2 = batch_x2.reshape(batch_x2.shape[0],batch_x2.shape[1])
	                batch_y = batch_y.reshape(batch_y.shape[0],batch_y.shape[1])
	                l1,l2 = session.run([optimiser,totalLoss], feed_dict={input1: batch_x,outputPrev: batch_x2,output: batch_y})
	                t_loss = t_loss+l2
	        print "en = ",en,"it =",n_it
	        en = en+step_length
	    predictTestValues(session,"predictions_exp6"+str(n_it)+".csv")
