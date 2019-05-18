from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops, math_ops

import sru
import fru
import AdaptRegression as ar
import ARU as aru
from walmart_sales_utils import getWalmartEmbeddingTransform
from electricity_utils import getEmbeddingElectricity
from trafficPEMS_utils import getEmbeddingTraffic
from RossmanUtils import transformInput
from oneC_utils import getEmbeddingOneC
from temporal_conv_utils import temporal_convolution_layer


class Models():
    def __init__(self, params, inputPh, lagPh, outputPh, maxYYPh, tsIndicesPh, maskPh,
                 keep_prob, alpha_list,
                 mode, numFW, aruDims, feature_dict, emb_dict, numTs):
        self.inputPh = inputPh
        self.lagPh = lagPh
        self.outputPh = outputPh
        self.maxYYPh = maxYYPh
        self.tsIndicesPh = tsIndicesPh
        self.maskPh = maskPh
        self.feature_dict = feature_dict
        self.emb_dict = emb_dict
        self.rnnType = params.rnnType
        self.decoderType = params.decoderType
        self.isSRUFeat = params.isSRUFeat
        self.isLagFeat = params.isLagFeat
        self.num_layers = params.num_layers
        self.encoder_length = params.encoder_length
        self.sequence_length = params.sequence_length
        self.decoder_length = params.decoder_length
        self.lstm_dense_layer_size = params.lstm_dense_layer_size
        self.hidden1 = params.hidden1
        self.hidden2 = params.hidden2
        self.keep_prob = keep_prob
        self.sru_num_stats = params.sru_num_stats
        self.alpha_list = alpha_list
        self.recur_dims = params.recur_dims
        self.num_nodes = params.num_nodes
        self.isNormalised = params.isNormalised
        self.deep_ar_normalize = params.deep_ar_normalize
        self.aru_deepar_normalize = params.aru_deepar_normalize
        self.mode = mode
        self.numFW = numFW
        self.step_length = params.stepLen
        self.seed = params.seed
        self.num_projs = params.numProjs
        self.fullLinear = params.fullLinear
        self.dataset = params.dataset
        self.numY = params.numY
        self.modelToRun = params.modelToRun
        self.aruDims = aruDims
        self.aruRegularizer = params.aruRegularizer
        self.is_maru_loss = params.is_maru_loss
        self.is_probabilistic_pred = params.is_probabilistic_pred
        self.aru_alpha = params.aru_alpha
        self.numTs = numTs

        self.maskPh = tf.squeeze(tf.cast(self.maskPh, tf.bool), axis=1)
        self.inputPh = tf.boolean_mask(self.inputPh, self.maskPh)
        self.lagPh = tf.boolean_mask(self.lagPh, self.maskPh)
        self.outputPh = tf.boolean_mask(self.outputPh, self.maskPh)
        self.maxYYPh = tf.boolean_mask(self.maxYYPh, self.maskPh)
        self.tsIndicesPh = tf.boolean_mask(self.tsIndicesPh, self.maskPh)

        self.inputPhEmbd = self.getEmbeddingTransform()
        self.inputEnc = tf.concat([self.inputPhEmbd[:, :self.encoder_length, :],
                                   self.lagPh[:, :self.encoder_length, :]],
                                   axis=2)
        self.inputDec = tf.concat([self.inputPhEmbd[:, self.encoder_length:, :],
                                   self.lagPh[:, self.encoder_length:, 1:]], # inputDec shouldn't get prevY, which is at lagPh[:, :, 0]
                                   axis=2)
        self.outputEnc = self.outputPh[:, :self.encoder_length, 0]
        self.outputDec = self.outputPh[:, self.encoder_length:self.sequence_length, 0]


    def getEmbeddingTransform(self):
        
        inputPhEmbd = []
        for feature, idx in self.feature_dict.items():
            #print(feature, idx, feature in self.emb_dict.keys())
            if feature not in self.emb_dict.keys():
                inputPhEmbd.append(self.inputPh[:, :, idx:idx+1])
            else:
                num_vals, emb_size = self.emb_dict[feature]
                v_index = tf.Variable(tf.random_uniform([num_vals, emb_size], -1.0, 1.0, seed=12))
                #v_index = tf.Print(v_index, [self.inputPh[:, :, idx]])
                em_index = tf.nn.embedding_lookup(v_index, tf.cast(self.inputPh[:, :, idx], tf.int32))
                inputPhEmbd.append(em_index)

        if self.inputPh.get_shape().as_list()[2] == 0:
            X_embd = self.inputPh
        else:
            X_embd = tf.concat(inputPhEmbd, axis=2)
        return X_embd

    def sru_cell(self):
        return sru.SimpleSRUCell(num_stats=self.sru_num_stats, mavg_alphas=tf.get_variable('alphas',
                                 initializer=tf.constant(self.alpha_list), trainable=False),
                                 output_dims=self.num_nodes, recur_dims=self.recur_dims)

    #LSTM Cell
    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.num_nodes, forget_bias=1.0)

    #Kind of Cell required
    def get_cell(self):
        if self.rnnType == "lstm":
            #print ("getting lstm")
            return self.lstm_cell()
        elif self.rnnType == "sru":
            #print ("getting sru")
            return self.sru_cell()

    def expMovAvg(self, y, alpha):
        expAvg = list()
        expAvg.append(tf.expand_dims(y[:,0], axis=1))
        for i in range(1, y.get_shape().as_list()[1]):
            avg = alpha * expAvg[-1] + (1 - alpha) * tf.expand_dims(y[:, i], axis=1)
            expAvg.append(avg)
        expAvg = tf.concat(expAvg, axis=1)
        return expAvg

    def multiScaleAverage(self, outputEncN):
        avgOutputs = list()
        for alpha in self.alpha_list:
            avgOutput = self.expMovAvg(outputEncN, alpha)
            avgOutputs.append(tf.expand_dims(avgOutput, axis=2))
        avgOutputs = tf.concat(avgOutputs, axis=2)
        return avgOutputs

    def encodeInput(self):
            if self.rnnType in ['lstm']:
                self.cell = tf.contrib.rnn.MultiRNNCell([self.get_cell() for _ in range(self.num_layers)],
                                                    state_is_tuple=True)
                state = self.cell.zero_state(tf.shape(self.inputEnc)[0], dtype = tf.float32)
            elif self.rnnType in ['cudnn_lstm']:
                self.cell = cudnn_rnn.CudnnLSTM(num_layers=self.num_layers, num_units=self.num_nodes,
                                           kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
                                           direction='unidirectional',
                                           seed=self.seed)
            elif self.rnnType in ['feed_forward']:
                inputs = self.inputEnc
                for l in range(self.num_layers):
                    inputs = tf.layers.dense(inputs, self.num_nodes, activation=tf.nn.relu)
                cell_outputs, state = inputs, None


            if self.rnnType in ['lstm']:
                cell_outputs = list()
                for i in range(self.encoder_length):
                    cell_output, state = self.cell(self.inputEnc[:, i, :],  state)
                    cell_outputs.append(tf.expand_dims(cell_output, axis=1))
                cell_outputs = tf.concat(cell_outputs, axis=1)
            elif self.rnnType in ['cudnn_lstm']:
                cell_outputs, (h_state, c_state) = self.cell(tf.transpose(self.inputEnc, [1, 0, 2]))
                cell_outputs = tf.transpose(cell_outputs, [1, 0, 2])
                state = (h_state, c_state)
            elif self.rnnType in ['feed_forward']:
                pass

            return cell_outputs, state

    def decoder(self):
            self.decoder_cell = cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.hidden1,
                                                    kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
                                                    direction='unidirectional',
                                                    seed=self.seed)

            decoder_input = tf.concat([self.inputPhEmbd, self.lagPh[:, :, 1:]], axis=1)
            decoder_outputs, (h_state, c_state) = self.decoder_cell(tf.transpose(decoder_input, [1, 0, 2]))
            decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])

            return decoder_outputs, h_state, c_state


    def nn_pass(self, inputS):
            with tf.variable_scope('nn_pass', reuse=tf.AUTO_REUSE):
                nn_input = tf.layers.dense(inputS, self.hidden1, activation=tf.nn.relu, name='hidden1_layer',
                                           kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                #nn_input = tf.nn.dropout(nn_input, self.keep_prob, name='dropout_layer', seed=self.seed)
                nn_input = tf.layers.dense(nn_input, self.hidden2, activation=tf.nn.relu, name='hidden2_layer',
                                           kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                y_pred_nn = tf.layers.dense(inputs=nn_input, units=1, activation=None, name='Ouptput_layer',
                                            kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                return nn_input, y_pred_nn


    def getARUModel(self):
            tf.set_random_seed(self.seed)

            if self.deep_ar_normalize:
                #Normalise y
                avg_output = tf.expand_dims((1 + tf.reduce_mean(self.outputDec, axis=1)), axis=1)
                outputDecN = self.outputDec/avg_output
            else:
                outputDecN = self.outputDec

            if self.isSRUFeat:
                self.sru_states = self.multiScaleAverage(self.inputEnc[:, :, 0])
                sru_state = tf.tile(tf.expand_dims(self.sru_states[:, -1, :], dim=1), [1, self.decoder_length, 1])

            cell_outputs, state = self.encodeInput()
            cell_output = cell_outputs[:, -1, :] if self.rnnType not in ['feed_forward'] else cell_outputs
            lstm_out_dense = tf.layers.dense(inputs=cell_output,
                                             units=self.lstm_dense_layer_size,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
            if self.rnnType not in ['feed_forward']:
                lstm_out_dense = tf.tile(tf.expand_dims(lstm_out_dense, dim=1), [1, self.decoder_length, 1])

            adapt_regression = aru.ARU(self.aruDims,
                                       self.numTs,
                                       num_y=self.numY, num_projs=self.num_projs,
                                       full_linear=self.fullLinear,
                                       regularizer=self.aruRegularizer,
                                       proj_vecs=None,
                                       aru_alpha=self.aru_alpha)
            if self.modelToRun in ['enc_aru']:
                adapt_input = self.inputPh[:, self.encoder_length-self.step_length:self.encoder_length, :]
                adapt_labels = tf.expand_dims(self.outputEnc[:, self.encoder_length-self.step_length:self.encoder_length], 2)
                #adapt_input = self.inputPh[:, :self.encoder_length, :]
                #adapt_labels = tf.expand_dims(self.outputEnc, 2)
            elif self.modelToRun in ['dec_aru']:
                adapt_input = self.inputPh[:, self.encoder_length:self.encoder_length+self.step_length, :]
                adapt_labels = tf.expand_dims(self.outputDec[:, :self.step_length], 2)

            output_h, _ = adapt_regression.predict(self.inputPh[:, self.encoder_length:, :], self.tsIndicesPh, self.maskPh)
            #output_h = tf.Print(output_h, [tf.reduce_sum(tf.cast(tf.is_nan(output_h), tf.int32))])

            if self.aru_deepar_normalize:
                avg_aru_output = tf.expand_dims((1 + tf.reduce_mean(output_h, axis=1)), axis=1)
                output_h = output_h / avg_aru_output

            with tf.control_dependencies([output_h]):
                adapted_states = adapt_regression.adapt(adapt_input, adapt_labels, self.tsIndicesPh, self.maskPh)

            #output_h = tf.Print(output_h, [tf.reduce_sum(tf.cast(tf.is_nan(self.inputPhEmbd), tf.int32))])
            nn_input = tf.concat([self.inputDec, lstm_out_dense, output_h], axis=2)
            if self.isSRUFeat:
                nn_input = tf.concat([nn_input, sru_state], axis=2)

            nn_input, y_pred_nn = self.nn_pass(nn_input)
            y_pred_nn = tf.squeeze(y_pred_nn, [2])
            y_pred_nn = y_pred_nn# * self.maskPh
            #outputDecN *= self.maskPh

            if self.is_probabilistic_pred:
                mu = y_pred_nn
                sigma = tf.layers.dense(inputs=nn_input, units=1, activation=tf.nn.softplus,
                                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                sigma = tf.squeeze(sigma, [2])
                dist = tfd.Normal(loc=mu, scale=sigma)
                log_likelihood = dist.log_prob(outputDecN)
                #log_likelihood *= self.maskPh
                loss = -1.0 * tf.reduce_sum(log_likelihood)
                self.sigma = sigma
            else:
                loss = tf.reduce_sum(tf.square(y_pred_nn - outputDecN))

            if self.deep_ar_normalize:
                #Rescale y
                y_pred_nn = y_pred_nn*avg_output

            self.reset_op = adapt_regression.reset_states()

            self.loss = loss
            self.y_pred_nn = y_pred_nn
            return loss, y_pred_nn, adapted_states


    def getModifiedARUModel(self):
            tf.set_random_seed(self.seed)

            if self.deep_ar_normalize:
                #Normalise y
                avg_output = tf.expand_dims((1 + tf.reduce_mean(self.outputDec, axis=1)), axis=1)
                outputDecN = self.outputDec/avg_output
                avg_enc_output = tf.expand_dims((1 + tf.reduce_mean(self.outputEnc, axis=1)), axis=1)
                outputEncN = self.outputEnc/avg_enc_output
            else:
                outputDecN = self.outputDec
                outputEncN = self.outputEnc

            cell_outputs, state = self.encodeInput() # encoder pass.
            cell_output = cell_outputs[:, -1, :] if self.rnnType not in ['feed_forward'] else cell_outputs

            lstm_out_dense = tf.layers.dense(inputs=cell_output,
                                             units=self.lstm_dense_layer_size,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
            if self.rnnType not in ['feed_forward']:
                lstm_out_dense = tf.tile(tf.expand_dims(lstm_out_dense, axis=1), [1, self.decoder_length, 1])
            if self.isSRUFeat or self.modelToRun in ['maruX_sruAugment']:
                sru_state = self.multiScaleAverage()
                sru_state = tf.tile(tf.expand_dims(sru_state, dim=1), [1, self.decoder_length, 1])

            if self.modelToRun in ['maruY']:
                with tf.variable_scope('y_decomposition', reuse=tf.AUTO_REUSE):
                    y_latent = tf.layers.dense(tf.expand_dims(self.outputDec, axis=2), self.numY, activation=tf.nn.relu)
                    # Add a dropout layer during training
                    y_latent = tf.cond(tf.equal(self.mode, 1.0),
                                       lambda: tf.nn.dropout(y_latent, 0.8, name='dropout_layer'),
                                       lambda: y_latent)
                    y_reconst = tf.layers.dense(y_latent, 1, activation=None, name='reconst_layer')

            # Fix adapt_input
            if self.modelToRun in ['maruY']:
                adapt_input = self.inputPh[:,self.encoder_length:self.encoder_length+self.step_length, :]
                aruInput = self.inputPh[:, self.encoder_length:, :]
            elif self.modelToRun in ['maruX_sruAugment']:
                adapt_input = tf.concat([sru_state[:, self.encoder_length:self.encoder_length+self.step_length, :],
                                        self.inputPh[:, self.encoder_length:self.encoder_length+self.step_length, :]],
                                        axis=2)
                aruInput = self.inputPh[:, self.encoder_length:, :]
            elif self.modelToRun in ['maruX_rnnAugment']:
                adapt_input = tf.concat([lstm_out_dense[:, self.encoder_length:self.encoder_length+self.step_length, :],
                                        self.inputPh[:, self.encoder_length:self.encoder_length+self.step_length, :]],
                                        axis=2)
                aruInput = self.inputPh[:, self.encoder_length:, :]
            elif self.modelToRun in ['maruX']:
                def nn_pass_maruX(nn_input):
                    with tf.variable_scope('nn_pass_maruX', reuse=tf.AUTO_REUSE):
                        nn_input = tf.layers.dense(nn_input, self.hidden1, activation=tf.nn.relu,
                                                   kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                        adapt_input = tf.layers.dense(nn_input, self.hidden2, activation=tf.nn.relu,
                                                      kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                    return adapt_input
                nn_input = self.inputPh[:, self.encoder_length:self.encoder_length+self.step_length, :]
                adapt_input = nn_pass_maruX(nn_input)
                aruInput = self.inputPh[:, self.encoder_length:, :]
                aruInput = nn_pass_maruX(aruInput)
            # Fix adapt_output
            if self.modelToRun in ['maruY']:
                adapt_labels = y_latent[:, :self.step_length, :]
            elif self.modelToRun in ['maruX_sruAugment', 'maruX_rnnAugment', 'maruX']:
                adapt_labels = tf.expand_dims(self.outputDec[:, :self.step_length],  axis=2)

            adapt_regression = aru.ARU(self.aruDims,
                                       self.numTs,
                                       num_y=self.numY,
                                       num_projs=self.num_projs,
                                       full_linear=self.fullLinear,
                                       regularizer=self.aruRegularizer,
                                       aru_alpha=self.aru_alpha)
            output_h, _ = adapt_regression.predict(aruInput, self.tsIndicesPh, self.maskPh)
            if self.aru_deepar_normalize:
                avg_aru_output = tf.expand_dims((1 + tf.reduce_mean(output_h, axis=1)), axis=1)
                output_h = output_h / avg_aru_output
            with tf.control_dependencies([output_h]):
                adapted_states = adapt_regression.adapt(adapt_input, adapt_labels, self.tsIndicesPh, self.maskPh)

            nn_input = tf.concat([self.inputDec, lstm_out_dense, output_h], axis=2)
            if self.modelToRun in ['maruX']:
                nn_input = tf.concat([nn_input, aruInput], axis=2)
            nn_input, y_pred_nn = self.nn_pass(nn_input)
            y_pred_nn = tf.squeeze(y_pred_nn,[2])
            y_pred_nn = y_pred_nn# * self.maskPh
            #outputDecN *= self.maskPh

            if self.is_probabilistic_pred:
                mu = y_pred_nn
                sigma = tf.layers.dense(inputs=nn_input, units=1, activation=tf.nn.softplus,
                                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                sigma = tf.squeeze(sigma, [2])
                dist = tfd.Normal(loc=mu, scale=sigma)
                log_likelihood = dist.log_prob(outputDecN)
                #log_likelihood *= self.maskPh
                loss = -1.0 * tf.reduce_sum(log_likelihood)
                self.sigma = sigma
            else:
                loss = tf.reduce_sum(tf.square(y_pred_nn - outputDecN))

            if self.is_maru_loss:
                y_pred_global = tf.layers.dense(inputs=aruInput, units=1, activation=None, name='globalOutputLayer',
                                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                loss += tf.reduce_sum(tf.square(tf.squeeze(y_pred_global, axis=2) - outputDecN))
            if self.modelToRun in ['maruY']:
                loss += tf.reduce_sum(tf.square(tf.squeeze(y_reconst, axis=2) - self.outputDec))

            if self.deep_ar_normalize:
                #Rescale y
                y_pred_nn = y_pred_nn*avg_output

            self.reset_op = adapt_regression.reset_states()
            self.loss = loss
            self.y_pred_nn = y_pred_nn
            return loss, y_pred_nn, adapted_states

    def getModifiedEncARUModel(self):
            tf.set_random_seed(self.seed)

            if self.deep_ar_normalize:
                #Normalise y
                avg_output = tf.expand_dims((1 + tf.reduce_mean(self.outputDec, axis=1)), axis=1)
                outputDecN = self.outputDec/avg_output
                avg_enc_output = tf.expand_dims((1 + tf.reduce_mean(self.outputEnc, axis=1)), axis=1)
                outputEncN = self.outputEnc/avg_enc_output
            else:
                outputDecN = self.outputDec
                outputEncN = self.outputEnc

            cell_outputs, state = self.encodeInput() # encoder pass.
            cell_output = cell_outputs if self.rnnType not in ['feed_forward'] else cell_outputs
            lstm_out_dense = tf.layers.dense(inputs=cell_output,
                                             units=self.lstm_dense_layer_size,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
            if self.rnnType not in ['feed_forward']:
                lstm_out_dense = tf.tile(tf.expand_dims(lstm_out_dense[:,-1,:], axis=1), [1, self.decoder_length, 1])
            if self.isSRUFeat or self.modelToRun in ['mEncAruX_sruAugment']:
                sru_state = self.multiScaleAverage(outputEncN)
                #sru_state = tf.tile(tf.expand_dims(sru_state, dim=1), [1, self.decoder_length, 1])

            if self.modelToRun in ['mEncAruY']:
                with tf.variable_scope('y_decomposition', reuse=tf.AUTO_REUSE):
                    y_latent = tf.layers.dense(tf.expand_dims(self.outputEnc, axis=2), self.numY, activation=tf.nn.relu)
                    # Add a dropout layer during training
                    y_latent = tf.cond(tf.equal(self.mode, 1.0),
                                       lambda: tf.nn.dropout(y_latent, 0.7, name='dropout_layer'),
                                       lambda: y_latent)
                    y_reconst = tf.layers.dense(y_latent, 1, activation=None, name='reconst_layer')

            # Fix adapt_input
            if self.modelToRun in ['mEncAruX']:
                nn_input = self.inputPh[:,:self.step_length,:]
                def nn_pass_mEncAruX(nn_input):
                    with tf.variable_scope('nn_pass_mEncAruX', reuse=tf.AUTO_REUSE):
                        nn_input = tf.layers.dense(nn_input, self.hidden1, activation=tf.nn.relu,
                                                   kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                        adapt_input = tf.layers.dense(nn_input, self.hidden2, activation=tf.nn.relu,
                                                   kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                    return adapt_input
                adapt_input = nn_pass_mEncAruX(nn_input)
                aruInput = self.inputPh[:,self.encoder_length:,:]
                aruInput = nn_pass_mEncAruX(aruInput)
            elif self.modelToRun in ['mEncAruY']:
                adapt_input = self.inputPh[:,:self.step_length,:]
                aruInput = self.inputPh[:,self.encoder_length:,:]
            elif self.modelToRun in ['mEncAruX_sruAugment']:
                adapt_input = tf.concat([sru_state[:,:self.step_length,:], self.inputPh[:,:step_length,:]], axis=2)
                aruInput = tf.concat([tf.tile(tf.expand_dims(sru_state[:,-1,:], axis=1),
                                              [1, self.decoder_length, 1]),
                                     self.inputPh[:,self.encoder_length:,:]], axis=2)
            elif self.modelToRun in ['mEncAruX_rnnAugment']:
                adapt_input = tf.concat([lstm_out_dense, self.inputPh[:,:self.step_length,:]], axis=2)
                aruInput = tf.concat([lstm_out_dense, self.inputPh[:,self.encoder_length:,:]], axis=2)
            # Fix adapt_output
            if self.modelToRun in ['mEncAruY']:
                adapt_labels = y_latent[:,:self.step_length,:]
            elif self.modelToRun in ['mEncAruX_sruAugment','mEncAruX_rnnAugment','mEncAruX']:
                adapt_labels = tf.expand_dims(self.outputEnc[:,:self.step_length], axis=2)

            adapt_regression = aru.ARU(self.aruDims,
                                       self.numTs,
                                       num_y=self.numY,
                                       num_projs=self.num_projs,
                                       full_linear=self.fullLinear,
                                       regularizer=self.aruRegularizer,
                                       aru_alpha=self.aru_alpha)
            output_h,_ = adapt_regression.predict(aruInput, self.tsIndicesPh, self.maskPh)
            if self.aru_deepar_normalize:
                avg_aru_output = tf.expand_dims((1 + tf.reduce_mean(output_h, axis=1)), axis=1)
                output_h = output_h / avg_aru_output

            with tf.control_dependencies([output_h]):
                adapted_states = adapt_regression.adapt(adapt_input, adapt_labels, self.tsIndicesPh, self.maskPh)

            nn_input = tf.concat([self.inputDec, lstm_out_dense, output_h], axis=2)
            if self.modelToRun in ['mEncAruX']:
                nn_input = tf.concat([nn_input, aruInput], axis=2)

            nn_input, y_pred_nn = self.nn_pass(nn_input)
            y_pred_nn = tf.squeeze(y_pred_nn,[2])
            y_pred_nn = y_pred_nn# * self.maskPh
            #outputDecN *= self.maskPh

            if self.is_probabilistic_pred:
                mu = y_pred_nn
                sigma = tf.layers.dense(inputs=nn_input, units=1, activation=tf.nn.softplus,
                                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                sigma = tf.squeeze(sigma, [2])
                dist = tfd.Normal(loc=mu, scale=sigma)
                log_likelihood = dist.log_prob(outputDecN)
                #log_likelihood *= self.maskPh
                loss = -1.0 * tf.reduce_sum(log_likelihood)
                self.sigma = sigma
            else:
                loss = tf.reduce_sum(tf.square(y_pred_nn - outputDecN))

            if self.modelToRun in ['mEncAruY']:
                loss += tf.reduce_sum(tf.square(tf.squeeze(y_reconst, axis=2) - self.outputEnc))

            if self.deep_ar_normalize:
                #Rescale y
                y_pred_nn = y_pred_nn*avg_output

            self.reset_op = adapt_regression.reset_states()
            self.loss = loss
            self.y_pred_nn = y_pred_nn
            return loss, y_pred_nn, adapted_states


    def getModifiedEncARUModelOld(self):
            tf.set_random_seed(self.seed)

            if self.deep_ar_normalize:
                #Normalise y
                avg_output = tf.expand_dims((1 + tf.reduce_mean(self.outputDec, axis=1)), axis=1)
                outputDecN = self.outputDec/avg_output
                avg_enc_output = tf.expand_dims((1 + tf.reduce_mean(self.outputEnc, axis=1)), axis=1)
                outputEncN = self.outputEnc/avg_enc_output
            else:
                outputDecN = self.outputDec
                outputEncN = self.outputEnc

            cell_outputs, state = self.encodeInput() # encoder pass.
            cell_output = cell_outputs
            lstm_out_dense = tf.layers.dense(inputs=cell_output,
                                             units=self.lstm_dense_layer_size,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
            lstm_out_dense = tf.tile(tf.expand_dims(lstm_out_dense[:,-1,:], axis=1), [1, self.decoder_length, 1])
            if self.isSRUFeat or self.modelToRun in ['mEncAruX_sruAugment']:
                sru_state = self.multiScaleAverage(outputEncN)
                #sru_state = tf.tile(tf.expand_dims(sru_state, dim=1), [1, self.decoder_length, 1])

            if self.modelToRun in ['mEncAruY']:
                with tf.variable_scope('y_decomposition', reuse=tf.AUTO_REUSE):
                    y_latent = tf.layers.dense(tf.expand_dims(self.outputEnc, axis=2), self.numY, activation=tf.nn.relu)
                    # Add a dropout layer during training
                    y_latent = tf.cond(tf.equal(self.mode, 1.0),
                                       lambda: tf.nn.dropout(y_latent, 0.7, name='dropout_layer'),
                                       lambda: y_latent)
                    y_reconst = tf.layers.dense(y_latent, 1, activation=None, name='reconst_layer')

            # Fix adapt_input
            if self.modelToRun in ['mEncAruX']:
                nn_input = self.inputPh[:,:self.step_length,:]
                def nn_pass_mEncAruX(nn_input):
                    with tf.variable_scope('nn_pass_mEncAruX', reuse=tf.AUTO_REUSE):
                        nn_input = tf.layers.dense(nn_input, self.hidden1, activation=tf.nn.relu,
                                                   kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                        adapt_input = tf.layers.dense(nn_input, self.hidden2, activation=tf.nn.relu,
                                                   kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                    return adapt_input
                adapt_input = nn_pass_mEncAruX(nn_input)
                aruInput = self.inputPh[:,self.encoder_length:,:]
                aruInput = nn_pass_mEncAruX(aruInput)
            elif self.modelToRun in ['mEncAruY']:
                adapt_input = self.inputPh[:,:self.step_length,:]
                aruInput = self.inputPh[:,self.encoder_length:,:]
            elif self.modelToRun in ['mEncAruX_sruAugment']:
                adapt_input = tf.concat([sru_state[:,:self.step_length,:], self.inputPh[:,:step_length,:]], axis=2)
                aruInput = tf.concat([tf.tile(tf.expand_dims(sru_state[:,-1,:], axis=1),
                                              [1, self.decoder_length, 1]),
                                     self.inputPh[:,self.encoder_length:,:]], axis=2)
            elif self.modelToRun in ['mEncAruX_rnnAugment']:
                adapt_input = tf.concat([lstm_out_dense, self.inputPh[:,:self.step_length,:]], axis=2)
                aruInput = tf.concat([lstm_out_dense, self.inputPh[:,self.encoder_length:,:]], axis=2)
            # Fix adapt_output
            if self.modelToRun in ['mEncAruY']:
                adapt_labels = y_latent[:,:self.step_length,:]
            elif self.modelToRun in ['mEncAruX_sruAugment','mEncAruX_rnnAugment','mEncAruX']:
                adapt_labels = tf.expand_dims(self.outputEnc[:,:self.step_length], axis=2)

            adapt_regression = aru.ARU(self.aruDims, self.numTs, num_y=self.numY, num_projs=self.num_projs,
                                       full_linear=self.fullLinear, regularizer=self.aruRegularizer)
            output_h,ar_state_tuple = adapt_regression.predict(aruInput, self.tsIndicesPh, self.maskPh)
            adapted_states = adapt_regression.adapt(adapt_input, adapt_labels, self.tsIndicesPh, self.maskPh)

            nn_input = tf.concat([self.inputDec, lstm_out_dense, output_h], axis=2)
            if self.modelToRun in ['mEncAruX']:
                nn_input = tf.concat([nn_input, aruInput], axis=2)

            _, y_pred_nn = self.nn_pass(nn_input)
            y_pred_nn = tf.squeeze(y_pred_nn,[2])
            y_pred_nn = y_pred_nn# * self.maskPh
            #outputDecN *= self.maskPh

            loss = tf.reduce_sum(tf.square(y_pred_nn - outputDecN))
            if self.modelToRun in ['mEncAruY']:
                loss += tf.reduce_sum(tf.square(tf.squeeze(y_reconst, axis=2) - self.outputEnc))

            if self.deep_ar_normalize:
                #Rescale y
                y_pred_nn = y_pred_nn*avg_output

            self.reset_op = adapt_regression.reset_states()
            self.loss = loss
            self.y_pred_nn = y_pred_nn
            return loss, y_pred_nn, ar_state_tuple


    def getFullyLocalModel(self):
            tf.set_random_seed(self.seed)
            if self.deep_ar_normalize:
                #Normalise y
                avg_output = tf.expand_dims((1 + tf.reduce_mean(self.outputDec, axis=1)), axis=1)
                outputDecN = self.outputDec/avg_output
                avg_enc_output = tf.expand_dims((1 + tf.reduce_mean(self.outputEnc, axis=1)), axis=1)
                outputEncN = self.outputEnc/avg_enc_output
            else:
                outputDecN = self.outputDec
                outputEncN = self.outputEnc

            adapt_regression = aru.ARU(self.aruDims,
                                       self.numTs,
                                       num_y=self.numY,
                                       num_projs=self.num_projs,
                                       full_linear=self.fullLinear,
                                       regularizer=self.aruRegularizer,
                                       aru_alpha=self.aru_alpha)
            if self.modelToRun in ['localEncAru']:
                adapt_input = self.inputPh[:, self.encoder_length-self.step_length:self.encoder_length, :]
                adapt_labels = tf.expand_dims(self.outputEnc[:, self.encoder_length-self.step_length:self.encoder_length], 2)
            elif self.modelToRun in ['localDecAru']:
                adapt_input = self.inputPh[:, self.encoder_length:self.encoder_length+self.step_length, :]
                adapt_labels = tf.expand_dims(self.outputDec[:, :self.step_length], 2)

            output_h, _ = adapt_regression.predict(self.inputPh[:, self.encoder_length:, :], self.tsIndicesPh, self.maskPh)
            #output_h = tf.Print(output_h, [ar_state_tuple[2]])
            adapted_states = adapt_regression.adapt(adapt_input, adapt_labels, self.tsIndicesPh, self.maskPh)

            y_pred_nn = output_h[:, :, 0]
            #y_pred_nn = tf.squeeze(y_pred_nn, axis=2)
            y_pred_nn = y_pred_nn# * self.maskPh
            #outputDecN *= self.maskPh
            loss = tf.reduce_sum(tf.square(y_pred_nn - outputDecN))

            if self.deep_ar_normalize:
                #Rescale y
                y_pred_nn = y_pred_nn*avg_output

            self.reset_op = adapt_regression.reset_states()

            self.loss = loss
            self.y_pred_nn = y_pred_nn
            return loss, y_pred_nn, adapted_states


    def getBaselineModel(self):
            tf.set_random_seed(self.seed)

            if self.deep_ar_normalize:
                #Normalise y
                avg_output = tf.expand_dims((1 + tf.reduce_mean(self.outputDec, axis=1)), axis=1)
                outputDecN = self.outputDec/avg_output
            else:
                outputDecN = self.outputDec

            cell_outputs, state = self.encodeInput()
            if self.rnnType in ['cudnn_lstm']:
                h_state, c_state = state
            else:
                encoder_state = state
            cell_output = cell_outputs[:, -1, :]

            log_likelihood = list()
            y_pred_nn = list()
            sigma_pred_nn = list()
            for i in range(self.decoder_length):
                with tf.variable_scope('lstm_out_dense', reuse=tf.AUTO_REUSE):
                    lstm_out_dense = tf.layers.dense(inputs=cell_output,
                                                     units=self.lstm_dense_layer_size,
                                                     activation=tf.nn.relu, name='lstm_out_dense',
                                                     kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                lstm_out_dense = tf.expand_dims(lstm_out_dense, axis=1)
                nn_input = tf.expand_dims(self.inputDec[:, i, :], axis=1)
                nn_input_concat = tf.concat([nn_input, lstm_out_dense],axis=2)
                nn_input_concat, y_pred_nn_ = self.nn_pass(nn_input_concat)
                y_pred_nn_ = tf.squeeze(y_pred_nn_, axis=2)
                y_pred_nn.append(y_pred_nn_)
                next_input = tf.cond(tf.equal(self.mode, 1.0),
                                     lambda: tf.concat([outputDecN[:, i:i+1], self.inputDec[:, i, :]], axis=1),
                                     lambda: tf.concat([y_pred_nn_, self.inputDec[:, i, :]], axis=1))

                if self.rnnType in ['lstm']:
                    cell_output, encoder_state = self.cell(next_input, encoder_state)
                elif self.rnnType in ['cudnn_lstm']:
                    cell_output, (h_state, c_state) = self.cell(tf.transpose(tf.expand_dims(next_input, axis=1), [1, 0, 2]),
                                                                initial_state=(h_state, c_state))
                    cell_output = tf.transpose(cell_output, [1, 0, 2])
                    cell_output = cell_output[:, -1, :]

                if self.is_probabilistic_pred:
                    mu = y_pred_nn_
                    with tf.variable_scope('sigma_layer', reuse=tf.AUTO_REUSE):
                        sigma = tf.layers.dense(inputs=nn_input_concat, units=1, activation=tf.nn.softplus,
                                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                        sigma = tf.squeeze(sigma, axis=2)
                    dist = tfd.Normal(loc=mu, scale=sigma)
                    log_likelihood_ = dist.log_prob(outputDecN[:, i])
                    log_likelihood.append(log_likelihood_)
                    sigma_pred_nn.append(sigma)

            y_pred_nn = tf.concat(y_pred_nn, 1)
            y_pred_nn = y_pred_nn# * self.maskPh
            #outputDecN *= self.maskPh
            if self.is_probabilistic_pred:
                sigma_pred_nn = tf.concat(tf.squeeze(sigma_pred_nn, axis=2), 1)
                log_likelihood = tf.concat(log_likelihood, 1)
                #log_likelihood *= self.maskPh
                loss = -1.0 * tf.reduce_sum(log_likelihood)
                self.sigma = sigma_pred_nn
            else:
                loss = tf.reduce_sum(tf.square(y_pred_nn - outputDecN))

            if self.deep_ar_normalize:
                    #Rescale y
                    y_pred_nn = y_pred_nn*avg_output
                    
            self.loss = loss
            self.y_pred_nn = y_pred_nn
            return loss, y_pred_nn


    def getHybridModel(self):
            tf.set_random_seed(self.seed)

            if self.deep_ar_normalize:
                #Normalise y
                avg_output = tf.expand_dims((1 + tf.reduce_mean(self.outputDec, axis=1)), axis=1)
                outputDecN = self.outputDec/avg_output
            else:
                outputDecN = self.outputDec

            cell_outputs, state = self.encodeInput()
            cell_output = cell_outputs[:, -1, :]
            lstm_out_dense = tf.layers.dense(inputs=cell_output,
                                             units=self.lstm_dense_layer_size,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
            lstm_out_dense = tf.tile(tf.expand_dims(lstm_out_dense, dim=1), [1, self.decoder_length, 1])

            if self.decoderType in ['rnn']:
                inputDec, _, _ = self.decoder()
                inputDec = inputDec[:, self.encoder_length:, :]
            else:
                inputDec = self.inputDec
            nn_input = tf.concat([inputDec, lstm_out_dense], axis=2)

            nn_input, y_pred_nn = self.nn_pass(nn_input)
            y_pred_nn = tf.squeeze(y_pred_nn, [2])
            #outputDecN *= self.maskPh
            y_pred_nn = y_pred_nn# * self.maskPh

            if self.is_probabilistic_pred:
                mu = y_pred_nn
                sigma = tf.layers.dense(inputs=nn_input, units=1, activation=tf.nn.softplus,
                                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                sigma = tf.squeeze(sigma, [2])
                dist = tfd.Normal(loc=mu, scale=sigma)
                log_likelihood = dist.log_prob(outputDecN)
                #log_likelihood *= self.maskPh
                loss = -1.0 * tf.reduce_sum(log_likelihood)
                self.sigma = sigma
            else:
                loss = tf.reduce_sum(tf.square(y_pred_nn - outputDecN))

            if self.deep_ar_normalize:
                y_pred_nn = y_pred_nn*avg_output

            self.loss = loss
            self.y_pred_nn = y_pred_nn
            return loss, y_pred_nn


    def getSNAILModel(self):
            tf.set_random_seed(self.seed)

            if self.deep_ar_normalize:
                #Normalise y
                avg_output = tf.expand_dims((1 + tf.reduce_mean(self.outputDec, axis=1)), axis=1)
                outputDecN = self.outputDec/avg_output
            else:
                outputDecN = self.outputDec

            def denseBlock(inputs, dilation_rate, num_filters, scope):
                xf = temporal_convolution_layer(inputs, num_filters, 2, causal=True, dilation_rate=dilation_rate, scope=scope+'/kernel_1')
                xg = temporal_convolution_layer(inputs, num_filters, 2, causal=True, dilation_rate=dilation_rate, scope=scope+'/kernel_2')
                activations = tf.multiply(tf.tanh(xf), tf.nn.sigmoid(xg))
                return tf.concat([activations, inputs], axis=2)

            def TCBlock(inputs, num_filters):
                seq_len = inputs.get_shape().as_list()[1]
                for i in range(int(np.floor(np.log2(seq_len)))):
                    inputs = denseBlock(inputs, [np.power(2, i)], num_filters, 'conv_'+str(i))
                return inputs

            def attentionBlock(inputs, key_size, value_size):
                keys = tf.layers.dense(inputs, key_size)
                query = tf.layers.dense(inputs, key_size)
                values = tf.layers.dense(inputs, value_size)
                logits = tf.einsum('bij,bkj->bik', query, keys)
                #logits = tf.Print(logits, [tf.shape(logits)])
                mask = tf.constant(np.tril(np.ones([inputs.shape.as_list()[1], inputs.shape.as_list()[1]])), dtype=tf.float32)
                logits = tf.multiply(logits, mask)
                probs = tf.nn.softmax(logits)
                read = tf.einsum('bij,bjk->bik', probs, values)
                return tf.concat([inputs, read], axis=2)

            inputs = denseBlock(self.inputEnc, [1], 4, 'conv_1')
            inputs = attentionBlock(inputs, 4, 8)
            inputs = denseBlock(inputs, [2], 4, 'conv_2')
            inputs = attentionBlock(inputs, 4, 8)
            #inputs = TCBlock(self.inputEnc, 16)
            #inputs = attentionBlock(inputs, 16, 32)
            context_vec = inputs[:,-1,:]

            context_vec = tf.tile(tf.expand_dims(context_vec, dim=1), [1, self.decoder_length, 1])
            nn_input = tf.concat([self.inputDec, context_vec], axis=2)
            nn_input, y_pred_nn = self.nn_pass(nn_input)
            y_pred_nn = tf.squeeze(y_pred_nn,[2])
            y_pred_nn = y_pred_nn# * self.maskPh
            #outputDecN *= self.maskPh

            if self.is_probabilistic_pred:
                mu = y_pred_nn
                sigma = tf.layers.dense(inputs=nn_input, units=1, activation=tf.nn.softplus,
                                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                sigma = tf.squeeze(sigma, [2])
                dist = tfd.Normal(loc=mu, scale=sigma)
                log_likelihood = dist.log_prob(outputDecN)
                #log_likelihood *= self.maskPh
                loss = -1.0 * tf.reduce_sum(log_likelihood)
                self.sigma = sigma
            else:
                loss = tf.reduce_sum(tf.square(y_pred_nn - outputDecN))

            if self.deep_ar_normalize:
                y_pred_nn = y_pred_nn*avg_output

            return loss, y_pred_nn


    def getNeuralARUModel(self):
            tf.set_random_seed(self.seed)

            if self.deep_ar_normalize:
                #Normalise y
                avg_output = tf.expand_dims((1 + tf.reduce_mean(self.outputDec, axis=1)), axis=1)
                outputDecN = self.outputDec/avg_output
            else:
                outputDecN = self.outputDec

            cell_outputs, state = self.encodeInput()
            cell_output = cell_outputs[:, -1, :]
            lstm_out_dense = tf.layers.dense(inputs=cell_output,
                                             units=self.lstm_dense_layer_size,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
            lstm_out_dense = tf.tile(tf.expand_dims(lstm_out_dense, dim=1), [1, self.decoder_length, 1])

            def ARUPass(nn_input):
                with tf.variable_scope('nn_pass_maruX', reuse=tf.AUTO_REUSE):
                    nn_input = tf.layers.dense(nn_input, self.hidden1, activation=tf.nn.relu,
                                               kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                    nn_input = tf.layers.dense(nn_input, self.hidden2, activation=tf.nn.relu,
                                               kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                    W = tf.layers.dense(nn_input, self.aruDims, activation=None,
                                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                    b = tf.layers.dense(nn_input, 1, activation=None,
                                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                return W, b
            W, b = ARUPass(self.inputPh[:, self.encoder_length:, :])
            #W = tf.Print(W, [tf.reduce_sum(tf.cast(tf.is_nan(self.inputPh), tf.float32)), tf.shape(self.inputPh)])
            output_h = tf.expand_dims(tf.reduce_sum(self.inputPh[:, self.encoder_length:, :] * W, axis=2), axis=2) + b

            if self.decoderType in ['rnn']:
                inputDec, _, _ = self.decoder()
                inputDec = inputDec[:, self.encoder_length:, :]
            else:
                inputDec = self.inputDec
            nn_input = tf.concat([inputDec, lstm_out_dense, output_h], axis=2)

            nn_input, y_pred_nn = self.nn_pass(nn_input)
            y_pred_nn = tf.squeeze(y_pred_nn, [2])
            #outputDecN *= self.maskPh
            y_pred_nn = y_pred_nn# * self.maskPh

            if self.is_probabilistic_pred:
                mu = y_pred_nn
                sigma = tf.layers.dense(inputs=nn_input, units=1, activation=tf.nn.softplus,
                                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                sigma = tf.squeeze(sigma, [2])
                dist = tfd.Normal(loc=mu, scale=sigma)
                log_likelihood = dist.log_prob(outputDecN)
                #log_likelihood *= self.maskPh
                loss = -1.0 * tf.reduce_sum(log_likelihood)
                self.sigma = sigma
            else:
                loss = tf.reduce_sum(tf.square(y_pred_nn - outputDecN))

            if self.deep_ar_normalize:
                y_pred_nn = y_pred_nn*avg_output

            self.loss = loss
            self.y_pred_nn = y_pred_nn
            return loss, y_pred_nn


    def getAdaptModel(self):
            tf.set_random_seed(self.seed)

            if self.deep_ar_normalize:
                #Normalise y
                avg_output = tf.expand_dims((1 + tf.reduce_mean(self.outputDec, axis=1)), axis=1)
                outputDecN = self.outputDec/avg_output
            else:
                outputDecN = self.outputDec

            if self.isSRUFeat:
                self.sru_states = self.multiScaleAverage(self.inputEnc[:, :, 0])
                sru_state = tf.tile(tf.expand_dims(self.sru_states[:, -1, :], dim=1), [1, self.decoder_length, 1])

            cell_outputs, state = self.encodeInput()
            cell_output = cell_outputs[:, -1, :]
            lstm_out_dense = tf.layers.dense(inputs=cell_output,
                                             units=self.lstm_dense_layer_size,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
            lstm_out_dense = tf.tile(tf.expand_dims(lstm_out_dense, dim=1), [1, self.decoder_length, 1])

            adapt_regression = aru.ARU(self.aruDims,
                                       self.numTs,
                                       num_y=self.numY, num_projs=self.num_projs,
                                       full_linear=self.fullLinear,
                                       regularizer=self.aruRegularizer,
                                       proj_vecs=None,
                                       aru_alpha=self.aru_alpha)

            adapt_input = lstm_out_dense#[:, self.encoder_length:self.encoder_length+self.step_length, :]
            adapt_labels = tf.expand_dims(self.outputDec, 2)

            nn_input = tf.concat([self.inputDec, lstm_out_dense], axis=2)
            if self.isSRUFeat:
                nn_input = tf.concat([nn_input, sru_state], axis=2)

            nn_input, y_pred_global = self.nn_pass(nn_input)

            output_h, _ = adapt_regression.predict(nn_input, self.tsIndicesPh, self.maskPh)
            #output_h = tf.Print(output_h, [tf.reduce_sum(tf.cast(tf.is_nan(output_h), tf.int32))])

            if self.aru_deepar_normalize:
                avg_aru_output = tf.expand_dims((1 + tf.reduce_mean(output_h, axis=1)), axis=1)
                output_h = output_h / avg_aru_output

            with tf.control_dependencies([output_h]):
                adapted_states = adapt_regression.adapt(nn_input, adapt_labels, self.tsIndicesPh, self.maskPh)

            #output_h = tf.Print(output_h, [tf.reduce_sum(tf.cast(tf.is_nan(self.inputPhEmbd), tf.int32))])
            count = tf.gather(adapt_regression.count, self.tsIndicesPh)
            gate = tf.expand_dims(tf.expand_dims(tf.maximum(tf.sigmoid(count - 10),0.0), axis=1), axis=2)
            y_pred_nn = (1-gate)*y_pred_global + gate*output_h
            y_pred_nn = tf.squeeze(y_pred_nn, [2])
            y_pred_nn = y_pred_nn# * self.maskPh
            #outputDecN *= self.maskPh

            if self.is_probabilistic_pred:
                mu = y_pred_nn
                sigma = tf.layers.dense(inputs=nn_input, units=1, activation=tf.nn.softplus,
                                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                sigma = tf.squeeze(sigma, [2])
                dist = tfd.Normal(loc=mu, scale=sigma)
                log_likelihood = dist.log_prob(outputDecN)
                #log_likelihood *= self.maskPh
                loss = -1.0 * tf.reduce_sum(log_likelihood)
                self.sigma = sigma
            else:
                loss = tf.reduce_sum(tf.square(y_pred_nn - outputDecN))

            if self.deep_ar_normalize:
                #Rescale y
                y_pred_nn = y_pred_nn*avg_output

            self.reset_op = adapt_regression.reset_states()

            self.loss = loss
            self.y_pred_nn = y_pred_nn
            return loss, y_pred_nn, adapted_states



    def getAdaptModel_old(self,adapt_tuple_placeholders):
            tf.set_random_seed(self.seed)

#            if self.deep_ar_normalize:
#                #Normalise y
#                avg_output = tf.expand_dims((1 + tf.reduce_mean(outputYN[:,0:self.encoder_length],axis=1)),axis=1)
#                outputYN = outputYN/avg_output

            cell_outputs, state = self.encodeInput()
            h_state, c_state = state
            cell_output = cell_outputs[:, -1, :]
            lstm_out_dense = tf.layers.dense(inputs=cell_output, units=self.lstm_dense_layer_size,
                    activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))

            adapt_regression = ar.AdaptiveRegression(self.hidden2)
            ar_state_tuple = adapt_tuple_placeholders

            lstm_out_dense = tf.tile(tf.expand_dims(lstm_out_dense, dim=1), [1, self.decoder_length, 1])

            nn_input = tf.concat([self.inputDec, lstm_out_dense], axis=2)

            with tf.variable_scope('nn_pass', reuse=tf.AUTO_REUSE):
                nn_input = tf.nn.dropout(nn_input, self.keep_prob, name='dropout_layer')
                nn_input = tf.layers.dense(nn_input, self.hidden1, activation=tf.nn.relu, name='hidden1_layer',
                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                nn_input = tf.layers.dense(nn_input, self.hidden2, activation=tf.nn.relu, name='hidden2_layer',
                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                y_pred_nn2 = tf.layers.dense(inputs=nn_input, units=1, activation=None, name='Ouptput_layer',
                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))

            y_pred_nn2 = tf.squeeze(y_pred_nn2,[2])

            adapt_state = ar_state_tuple
            y_pred_nn, adapt_state = adapt_regression.predict(adapt_state, nn_input, y_pred_nn2)
            adapt_input = nn_input[:,self.decoder_length-self.step_length:self.decoder_length,:]
            adapt_labels = tf.expand_dims(self.outputDec[:,self.decoder_length-self.step_length:self.decoder_length], axis=2)
            adapt_state = adapt_regression.adapt(adapt_state, adapt_input, adapt_labels)
            loss = tf.reduce_sum(tf.square(y_pred_nn - self.outputDec))

            if self.deep_ar_normalize:
                #Rescale y
                y_pred_nn = y_pred_nn*avg_output

            return loss, y_pred_nn, adapt_state


    def getAttentionHybridModel(self):
            tf.set_random_seed(self.seed)
            inputS = self.inputPh
            
            if self.deep_ar_normalize:
                #Normalise y
                avg_output = tf.expand_dims((1 + tf.reduce_mean(self.outputDec, axis=1)), axis=1)
                outputDecN = self.outputDec/avg_output
                avg_enc_output = tf.expand_dims((1 + tf.reduce_mean(self.outputEnc, axis=1)), axis=1)
                outputEncN = self.outputEnc/avg_enc_output
            else:
                outputDecN = self.outputDec
                
            #lstm_input = tf.concat([inputS[:,0:self.encoder_length,:], 
            #    tf.expand_dims(self.prevy_p[:, 0:self.encoder_length], 2)], axis=2)
            
            #self.cell = tf.contrib.rnn.MultiRNNCell(
            #    [self.get_cell() for _ in range(self.num_layers)], state_is_tuple=True)
            #
            #state = self.cell.zero_state(tf.shape(self.prevy_p)[0], dtype = tf.float32)
            #
            #all_outputs = []
            #
            #for i in range(self.encoder_length):
            #    cell_output, state = self.cell(lstm_input[:, i, :], state)
            #    all_outputs.append(cell_output)
            #
            #all_outputs = tf.convert_to_tensor(all_outputs)
            #
            #trOut = tf.transpose(all_outputs, perm = [1, 0, 2])
            cell_outputs, state = self.encodeInput()
            cell_output = cell_outputs
            attentionMechanism = tf.contrib.seq2seq.BahdanauAttention(self.hidden1, cell_output,
                    memory_sequence_length=None,normalize=True,probability_fn=None,
                    score_mask_value=None,name='BahdanauAttention')
                
            #Y_out_i = tf.matmul(cell_output,Y_out_W_lstm)+Y_out_B_lstm
            lstm_out_dense = tf.layers.dense(inputs=cell_output[:,-1,:], units=self.lstm_dense_layer_size, 
                    activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
            next_attention_state = None
                
            predictions = []
            loss = 0
                
            #Same as the hybrid model where we predict all values in the decoder range, 
            #except we add attention also where we use the nn dense layer as a key
            for i in range(self.decoder_length):
                nn_input = self.inputDec[:,i,:]
                nn_input_concat = tf.concat([nn_input,lstm_out_dense],axis=1)
                nn_layer_1_dense = tf.layers.dense(inputs=nn_input_concat, units=self.hidden1, 
                        activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                # nn_layer_2_dense = tf.layers.dense(inputs=nn_layer_1_dense, units=hidden2, 
                #activation=tf.nn.relu)
                alignments, next_attention_state = attentionMechanism(nn_input_concat, 
                        state=next_attention_state)
                alignments = tf.expand_dims(alignments, axis=1)

                #2 different methods of getting the context
                # context = tf.matmul(alignments, attentionMechanism.values)
                context = tf.matmul(alignments, tf.expand_dims(self.outputEnc, axis=2))

                context = tf.squeeze(context, [1])
                nn_input_concat = tf.concat([nn_input_concat,context],axis=1)
                # nn_layer_1_dense = tf.layers.dense(inputs=nn_input_concat, units=hidden1, activation=tf.nn.relu)
                nn_layer_2_dense = tf.layers.dense(inputs=nn_input_concat, 
                        units=self.hidden2, activation=tf.nn.relu, 
                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                y_pred_nn = tf.layers.dense(inputs=nn_layer_2_dense, units=1, activation=None, 
                        kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
                #y_pred_nn = tf.squeeze(y_pred_nn, [2])
                var1 = tf.square(y_pred_nn - outputDecN[:,i])
                loss = loss+var1
                if self.deep_ar_normalize:
                    y_pred_nn = y_pred_nn*avg_output
                    predictions.append(y_pred_nn)
                else:
                    predictions.append(y_pred_nn)
                        
            predictions = tf.concat(predictions, axis=1)
            lossC = tf.reduce_sum(loss)
            return lossC, predictions


    def getHierarchicalModel(self):
                
                inputX = self.inputPh
                                
                def sumInput(inputX,start,end):
                        return tf.reduce_sum(inputX[:,start:end,:],1)
                
                def minInput(inputX,start,end):
                        return tf.reduce_min(inputX[:,start:end,:],1)
                
                def maxInput(inputX,start,end):
                        return tf.reduce_max(inputX[:,start:end,:],1)
                
                def meanInput(inputX,start,end):
                        return tf.reduce_mean(inputX[:,start:end,:],1)
                
                def aggregateOp(inputX,start,end):
                        return meanInput(inputX,int(start),int(end))
                
                def concat(X,prevY,prev_level_Y_list,prev_level_state_list,parent_index, lstm_out_dense):
                        concat_X = X
                        concat_X = tf.concat([concat_X,prevY],axis=1)
                        if len(prev_level_Y_list) > parent_index:
                                prev_level_y = prev_level_Y_list[parent_index]
                                prev_level_state = prev_level_state_list[parent_index]
                                concat_X = tf.concat([concat_X,prev_level_y],axis=1)
                                concat_X = tf.concat([concat_X,prev_level_state],axis=1)
                        elif len(prev_level_Y_list) > 0:
                                prev_level_y = prev_level_Y_list[-1]
                                prev_level_state = prev_level_state_list[-1]
                                concat_X = tf.concat([concat_X,prev_level_y],axis=1)
                                concat_X = tf.concat([concat_X,prev_level_state],axis=1)
                        concat_X = tf.concat([concat_X,lstm_out_dense],axis=1)
                        return concat_X
                        
                def one_nn_step(all_inputs,flag,name1,name2,name3):
                        if flag == True:
                                nn_input = tf.layers.dense(inputs=all_inputs, units=self.hidden1, activation=tf.nn.relu, name=name1, reuse=True)
                                nn_input = tf.layers.dense(inputs=all_inputs, units=self.hidden2, activation=tf.nn.relu, name=name2, reuse=True)
                                y_pred_nn = tf.layers.dense(inputs=nn_input, units=1, activation=None,name=name3,reuse=True)
                        else:
                                nn_input = tf.layers.dense(inputs=all_inputs, units=self.hidden1, activation=tf.nn.relu, name=name1)
                                nn_input = tf.layers.dense(inputs=all_inputs, units=self.hidden2, activation=tf.nn.relu, name=name2)
                                y_pred_nn = tf.layers.dense(inputs=nn_input, units=1, activation=None,name=name3)
                        return y_pred_nn,nn_input
                
                def lossC(y,nn_output,start_index,end_index):
                        y_out = aggregateOp(y,start_index,end_index)
                        loss = tf.reduce_sum(tf.abs(nn_output - y_out))
                        return loss

                #Encoding phase
                Y_out_W_lstm = tf.Variable(tf.random_uniform([self.num_nodes, self.lstm_dense_layer_size]))
                Y_out_B_lstm = tf.Variable(tf.random_uniform([self.lstm_dense_layer_size]))
                
                encoder_cell_output, encoder_state = self.encodeInput(inputX)

                # encoder_cell_output,encoder_state = encode_input()
                Y_out_i = tf.matmul(encoder_cell_output, Y_out_W_lstm) + Y_out_B_lstm
                lstm_out_dense = tf.layers.dense(inputs=Y_out_i, units=self.lstm_dense_layer_size, activation=tf.nn.relu)

                #lstm_out_dense = self.addSRUFeat(lstm_out_dense)

                #Decoding phase (We dont know Y)
                num_levels = int(np.log2(self.decoder_length))+1
                loss = 0
                prev_level_Y_list = []
                prev_level_state_list = []
                dictLevels = {}
                
                for level in range(num_levels):
                        l_flag = False
                        l_name1 = 'dense_layer_1'+str(level)
                        l_name2 = 'dense_layer_2'+str(level)
                        l_name3 = 'final_layer'+str(level)
                        agg_window = self.decoder_length/(2 ** level)
                        nodes_in_level = int(np.ceil(self.decoder_length/agg_window))
                        this_level_Y_list = []
                        this_level_state_list = []
                        prevY = tf.zeros([tf.shape(inputX)[0],1],dtype=tf.float32)
                        for s in range(nodes_in_level):
                                start_index = self.encoder_length + agg_window*s
                                end_index = min(start_index+agg_window, self.sequence_length)
                                parent_index = int(s/2)
                                X = aggregateOp(inputX,start_index,end_index) #sum,min,max,mean ??
                                all_inputs = concat(X, prevY, prev_level_Y_list, prev_level_state_list, parent_index, lstm_out_dense)
                                this_output, this_state = one_nn_step(all_inputs,l_flag,l_name1,l_name2,l_name3) #Use a nn layer instead of passing it through a lstm
                                l_flag = True
                                loss += lossC(self.outputPh, this_output, start_index, end_index)
                                prevY = this_output
                                this_level_Y_list.append(this_output)
                                this_level_state_list.append(this_state)
                        prev_level_Y_list = this_level_Y_list
                        prev_level_state_list = this_level_state_list
                        
                return loss, this_level_Y_list
