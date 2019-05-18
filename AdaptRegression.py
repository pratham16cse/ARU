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
import math

from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import sru
import fru
from AdaptUtil import *

ARStateTuple = collections.namedtuple("ARStateTuple", ("sxxt", "sxy", "count"))

class AdaptiveRegression(AdaptUtil):
  """LSTM unit with an adaptive linear regressor on the side.

  """

  def __init__(self, input_dims, num_y=1, adapt_threshold=5,
               dropout_keep_prob=1.0, dropout_prob_seed=None, regularizer=0.001,
               reuse=None):
    """Initializes the basic LSTM cell.

    Args:
      input_size: input size.
      dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
        recurrent dropout probability value. If float and 1.0, no dropout will
        be applied.
      dropout_prob_seed: (optional) integer, the randomness seed.
    """
    self._adapt_threshold = adapt_threshold
    self._reuse = reuse
    self._regularizer = regularizer
    self._input_dims = input_dims
    self._num_y = num_y


  def zero_state(self, batch_size, input_size):
    sxxt = tf.zeros([batch_size, input_size+1, input_size+1])
    sxy = tf.zeros([batch_size, input_size+1])
    count = tf.zeros([batch_size])
    return ARStateTuple(sxxt, sxy, count)

  def adapt(self, state, inputs, labels, valid=None):
    """
    :param inputs: [Batch,time,depth]
    :param state:  tuple*batch
    :param labels: [batch,time,num_y]
    :param valid: [batch,time,num_y]
    :return:
    """
    if state is None:
      state = self.zero_state(tf.shape(labels)[0], tf.shape(inputs)[2])
    
    sxxt, sxy, count = state
    # padding a constant to introduce a bias term.
    inputs = tf.concat([inputs, tf.ones([tf.shape(inputs)[0], tf.shape(inputs)[1], 1])], axis=2)

    if valid is not None:
      inputs = tf.cond(valid, inputs, tf.zeros_like(inputs))
      count += tf.reduce_sum(tf.cond(valid, 1, 0), axis=1)
    else:
      count += tf.to_float(tf.shape(labels)[1])
    xxt = tf.einsum('bti,btj->bij', inputs, inputs)
    xy = tf.einsum('bti,bty->biy', inputs, labels)
    sxxt += xxt
    sxy += xy
    return ARStateTuple(sxxt, sxy, count)


  def predict(self, state, inputs, default_output, labels=None, valid=None):
    """
    :param inputs: [Batch,time,depth]
    :param state:
    :param labels:
    :param valid:
    :return:
    """
    if state is None:
      return default_output, None

    W, b, count = self.get_Wb(state)
    gate = tf.expand_dims(tf.maximum(tf.sigmoid(count - self._adapt_threshold),0.0), 1)
    output = default_output
    output_local = (tf.einsum('btd,bd->bt', inputs, W) + b)
    output = (1-gate)*output + gate*output_local

    new_state = state
    if labels is not None:
      new_state = self.adapt(inputs, state, labels, valid)
    return output, new_state

  def get_Wb(self, state):
    if state is None:
      return None
    sxxt, sxy, count = state
    inverse_sxxt = tf.matrix_inverse(sxxt + self._regularizer * tf.expand_dims(tf.diag(tf.ones([tf.shape(sxy)[1]])), 0))
    #W = tf.squeeze(tf.matmul(inverse_sxxt, tf.expand_dims(sxy, 2)), axis=2)
    W = tf.squeeze(tf.matmul(inverse_sxxt, sxy), axis=2)
    W, b = tf.split(W, [tf.shape(W)[1] - 1, 1], axis=1)
    return W,b,count

  def print_state(self, state):
    if state is None:
      return
    W, b, c = self.get_Wb(state)
    tf.Print(W, [W,b,c], "Local W,b,c",)


  def getAdaptStateTuplePlaceholders(self):
    sxxt_p = tf.placeholder(tf.float32, [None,self._input_dims+1,self._input_dims+1])
    sxy_p = tf.placeholder(tf.float32, [None,self._input_dims+1, self._num_y])
    count_p = tf.placeholder(tf.float32,[None])
    self._num_states = 3
    return (sxxt_p,sxy_p,count_p)

  def getNeededStates(self,combined_state_tuple_current,k,combined_state_tuple_next):
    (sxxt_n,sxy_n,count_n) = combined_state_tuple_current
    (sxxt_needed,sxy_needed,count_needed) = combined_state_tuple_next
    sxxt_needed.append(sxxt_n[k,:,:])
    sxy_needed.append(sxy_n[k,:])
    count_needed.append(count_n[k])
    return (sxxt_needed,sxy_needed,count_needed)

  def initialize_zero_states(self,combined_state_tuple_n,hidden_dim):
    (sxxt_needed,sxy_needed,count_needed) = combined_state_tuple_n
    sxxt_needed.append(np.zeros((hidden_dim+1, hidden_dim+1)))
    sxy_needed.append(np.zeros((hidden_dim+1)))
    count_needed.append(0)
    return (sxxt_needed,sxy_needed,count_needed)

  def getZeroStateNumpyArray(self,batch_size_f,hidden2):
    sxxt_needed = np.zeros((batch_size_f,hidden2+1, hidden2+1))
    sxy_needed = np.zeros((batch_size_f,hidden2+1))
    count_needed = np.zeros((batch_size_f))
    return (sxxt_needed,sxy_needed,count_needed)
