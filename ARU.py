"""A recurrent unit that maintains many states of input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import tensorflow as tf
import numpy as np
from AdaptUtil import *

ARUStateTuple = collections.namedtuple("ARUStateTuple", ("sxxt", "sxy", "spx", "spxx", "sy", "spxy", "count"))

class ARU(AdaptUtil):
  """LSTM unit with an adaptive linear regressor on the side.

  """

  def __init__(self, input_dims, numTs, num_y=1, num_projs=10, full_linear=False, adapt_threshold=10, proj_vecs=None,
               dropout_keep_prob=1.0, dropout_prob_seed=None, regularizer=0.01, aru_alpha=None,
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
    self._nprojs=num_projs
    self._proj_vecs = proj_vecs
    self._input_dims = input_dims
    self._numTs = numTs
    self._num_y = num_y
    self._full_linear = full_linear
    self._alpha = aru_alpha
    self._num_alpha = len(self._alpha)
#    if self._nprojs > 0:
#        self._project_vectors = tf.truncated_normal(stddev=1.0, dtype=tf.float32, shape=[self._input_dims, self._nprojs], seed=12)
    if self._nprojs > 0:
      with tf.variable_scope(type(self).__name__):
          if self._proj_vecs is None:
              self._project_vectors = tf.Variable(tf.truncated_normal(stddev=1.0, dtype=tf.float32, shape=[self._input_dims, self._nprojs], seed=12), name="proj", trainable=False)
          else:
              self._project_vectors = tf.get_variable("proj", initializer=tf.constant(self._proj_vecs), trainable=False)

    self.sxxt = tf.Variable(tf.zeros([self._numTs, self._input_dims+1, self._input_dims+1, self._num_alpha]), trainable=False)
    self.sxy = tf.Variable(tf.zeros([self._numTs, self._input_dims+1, self._num_y, self._num_alpha]), trainable=False)

    self.spx2 = tf.Variable(tf.zeros([self._numTs, max(1, self._nprojs)]), trainable=False)
    self.spx = tf.Variable(tf.zeros([self._numTs, max(1, self._nprojs)]), trainable=False)
    self.spxy = tf.Variable(tf.zeros([self._numTs, max(1, self._nprojs), self._num_y]), trainable=False)
    self.sy = tf.Variable(tf.zeros([self._numTs, self._num_y]), trainable=False)
    self.count = tf.Variable(tf.zeros([self._numTs]), trainable=False)

  def zero_state(self, batch_size):
    sxxt = tf.zeros([batch_size, self._input_dims+1, self._input_dims+1])
    sxy = tf.zeros([batch_size, self._input_dims+1, self._num_y])
    
    spx2 = tf.zeros([batch_size, self._nprojs])
    spx = tf.zeros([batch_size, self._nprojs])
    spxy = tf.zeros([batch_size, self._nprojs, self._num_y])
    sy = tf.zeros([batch_size, self._num_y])
    
    count = tf.zeros([batch_size])
    return ARUStateTuple(sxxt, sxy, spx, spx2, sy, spxy, count)

  def reset_states(self):
    sxxt_assign_op = self.sxxt.assign(tf.zeros([self._numTs, self._input_dims+1, self._input_dims+1, self._num_alpha]))
    sxy_assign_op = self.sxy.assign(tf.zeros([self._numTs, self._input_dims+1, self._num_y, self._num_alpha]))
    spx2_assign_op = self.spx2.assign(tf.zeros([self._numTs, max(1, self._nprojs)]))
    spx_assign_op = self.spx.assign(tf.zeros([self._numTs, max(1, self._nprojs)]))
    spxy_assign_op = self.spxy.assign(tf.zeros([self._numTs, max(1, self._nprojs), self._num_y]))
    sy_assign_op = self.sy.assign(tf.zeros([self._numTs, self._num_y]))
    count_assign_op = self.count.assign(tf.zeros([self._numTs]))

    return ARUStateTuple(sxxt_assign_op, sxy_assign_op, spx_assign_op, spx2_assign_op, sy_assign_op, spxy_assign_op, count_assign_op)


  def adapt(self, inputs_arg, labels, tsIndices, mask, valid=None, scope=None):
    """
    :param inputs: [Batch,time,depth]
    :param state:  tuple*batch
    :param labels: [batch,time, num_y]
    :param valid: [batch,time]
    :return:
    """
    # print("in adapt state",tf.shape(state))
    
    #if state is None:
    #  state = self.zero_state(tf.shape(labels)[0])

    inputs_arg =  tf.stop_gradient(inputs_arg)
    #sxxt, sxy, spx, spx2, sy, spxy, count = state
    # padding a constant to introduce a bias term.
    inputs = tf.concat([inputs_arg, tf.ones([tf.shape(inputs_arg)[0], tf.shape(inputs_arg)[1], 1])], axis=2)
    if valid is not None:
      inputs = tf.cond(valid, inputs, tf.zeros_like(inputs))
      count_op = tf.scatter_add(self.count, tsIndices, tf.reduce_sum(tf.cond(valid, 1, 0), axis=1))#*tf.squeeze(mask))
    else:
      count_op = tf.scatter_add(self.count, tsIndices, tf.to_float(tf.shape(labels)[1]))#*tf.squeeze(mask))

    if self._full_linear:
      xxt = tf.einsum('bti,btj->bij', inputs, inputs)# * tf.expand_dims(mask, axis=1)
      xy = tf.einsum('bti,bty->biy', inputs, labels)# * tf.expand_dims(mask, axis=1)
      if self._alpha is None:
        sxxt_op = tf.scatter_add(self.sxxt, tsIndices, xxt)
        sxy_op = tf.scatter_add(self.sxy, tsIndices, xy)
      else:
        sxxt_mul = tf.scatter_mul(self.sxxt, tsIndices, tf.broadcast_to(self._alpha, [tf.shape(tsIndices)[0], self._input_dims+1, self._input_dims+1, self._num_alpha]))
        sxxt_op = tf.scatter_add(sxxt_mul, tsIndices, tf.tile(tf.expand_dims(xxt, axis=-1), [1, 1, 1, self._num_alpha]))
        sxy_mul = tf.scatter_mul(self.sxy, tsIndices, tf.broadcast_to(self._alpha, [tf.shape(tsIndices)[0], self._input_dims+1, self._num_y, self._num_alpha]))
        sxy_op = tf.scatter_add(sxy_mul, tsIndices, tf.tile(tf.expand_dims(xy, axis=-1), [1, 1, 1, self._num_alpha]))


#    if self._nprojs > 0:
#      with tf.variable_scope(scope or type(self).__name__):
#          if self._proj_vecs is None:
#              self._project_vectors = tf.Variable(tf.truncated_normal(stddev=1.0, dtype=tf.float32, shape=[self._input_dims, self._nprojs]), name="proj", trainable=False)
#          else:
#              self._project_vectors = tf.get_variable("proj", initializer=tf.constant_initializer(self._proj_vecs), trainable=False)

    if self._nprojs > 0:
      projected_inputs = tf.einsum('bti,ip->btp', inputs_arg, self._project_vectors)
      if self._alpha is None:
        spx_op = tf.scatter_add(self.spx, tsIndices, tf.reduce_sum(projected_inputs, 1))# * mask)
        spx2_op = tf.scatter_add(self.spx2, tsIndices, tf.reduce_sum(projected_inputs*projected_inputs, 1))# * mask)
        spxy_op = tf.scatter_add(self.spxy, tsIndices, tf.einsum('btp,bty->bpy', projected_inputs, labels))# * tf.expand_dims(mask, axis=1))
        sy_op = tf.scatter_add(self.sy, tsIndices, tf.reduce_sum(labels, 1))# * mask)
      else:
        spx_mul = tf.scatter_mul(self.spx, tsIndices, tf.ones((tf.shape(tsIndices)[0], max(1, self._nprojs)))*self._alpha)
        spx_op = tf.scatter_add(spx_mul, tsIndices, tf.reduce_sum(projected_inputs, 1))#*mask)
        spx2_mul = tf.scatter_mul(self.spx2, tsIndices, tf.ones((tf.shape(tsIndices)[0], max(1, self._nprojs)))*self._alpha)
        spx2_op = tf.scatter_add(spx2_mul, tsIndices, tf.reduce_sum((projected_inputs*projected_inputs, 1)))#*mask)
        spxy_mul = tf.scatter_mul(self.spxy, tsIndices, tf.ones((tf.shape(tsIndices)[0], max(1, self._nprojs), self._num_y))*self._alpha)
        spxy_op = tf.scatter_add(spxy_mul, tsIndices, tf.einsum('btp,bty->bpy', projected_inputs, labels))#*tf.expand_dims(mask, axis=1))
        sy_mul = tf.scatter_mul(self.sy, tsIndices, tf.ones((tf.shape(tsIndices)[0], self._num_y))*self._alpha)
        sy_op = tf.scatter_add(sy_mul, tsIndices, tf.reduce_sum(labels, 1))#*mask)

    else:
        spx_op = self.spx
        spx2_op = self.spx2
        spxy_op = self.spxy
        sy_op = self.sy

    return ARUStateTuple(sxxt_op, sxy_op, spx_op, spx2_op, sy_op, spxy_op, count_op)


  def predict(self, inputs, tsIndices, mask, labels=None, valid=None, scope=None):
    """
    :param inputs: [Batch,time,depth]
    :param state:
    :param labels:
    :param valid:
    :return output [Batch, time, (p+1)num_y]:
    """
    # print("in predict state",tf.shape(state))

    #if state is None:
    #  return None, None

    W, b, pW, pb, count,_,_ = self.get_Wb(tsIndices)
    output = None
    if self._full_linear:
      output = (tf.einsum('btd,bdya->btya', inputs, W) + b)

    if self._nprojs > 0:
        projected_inputs = tf.einsum('bti,ip->btp', inputs, self._project_vectors)
        output_proj = tf.einsum('btp,bpy->btpy', projected_inputs, pW) + tf.expand_dims(pb,1)
        # output = tf.concat([tf.expand_dims(output, 2), output_proj], 2)
        output_proj = tf.reshape(output_proj, [tf.shape(inputs)[0], inputs.get_shape().as_list()[1], -1])
        if output is not None:
          output = tf.concat([output, output_proj], 2)
          output = tf.reshape(output,[tf.shape(output)[0],tf.shape(output)[1],(self._nprojs+1)*self._num_y])
        else:
          output = output_proj
          output = tf.reshape(output,[tf.shape(output)[0],tf.shape(output)[1],(self._nprojs)*self._num_y])
    
    #new_state = state
    #if labels is not None:
      #new_state = self.adapt(inputs, state, labels, valid)
    output = output# * tf.expand_dims(mask, axis=1)
    output = tf.reshape(output, [tf.shape(output)[0], tf.shape(output)[1], self._num_y * self._num_alpha])
    return output, None

  def get_Wb(self, tsIndices):
    # if state is None:
    #   return None
    sxxt = tf.gather(self.sxxt, tsIndices)
    sxy = tf.gather(self.sxy, tsIndices)
    spx = tf.gather(self.spx, tsIndices)
    spx2 = tf.gather(self.spx2, tsIndices)
    sy = tf.gather(self.sy, tsIndices)
    spxy = tf.gather(self.spxy, tsIndices)
    count = tf.gather(self.count, tsIndices)

    #W = None
    #b = None
    if self._full_linear:
      #shape=[b,X+1,X+1]
      sxxt_ = tf.transpose(sxxt, perm=[0, 3, 1, 2])
      sxy_ = tf.transpose(sxy, perm=[0, 3, 1, 2])
      inverse_sxxt = tf.matrix_inverse(sxxt_ + self._regularizer * tf.expand_dims(tf.diag(tf.ones([tf.shape(sxxt)[1]])), 0))
      W = tf.einsum('bafx,baxy->bafy', inverse_sxxt, sxy_)
      W = tf.transpose(W, perm=[0, 2, 3, 1])
      W, b = tf.split(W, [self._input_dims, 1], axis=1)

    # [batch, nproj]
    pcount = tf.expand_dims(count,1)
    inverse_spx2 = pcount*spx2 - spx*spx + self._regularizer
    pW = (tf.expand_dims(pcount,2)*spxy - tf.einsum('bp,by->bpy',spx,sy))/tf.expand_dims(inverse_spx2, 2)
    pb = (tf.expand_dims(sy,1) - pW*tf.expand_dims(spx,2))/tf.expand_dims(pcount + self._regularizer, 2)
    return W,b,pW,pb,count,sxxt,sxy

  def print_state(self, state):
    if state is None:
      return
    W, b, c = self.get_Wb(state)
    tf.Print(W, [W,b,c], "Local W,b,c",)


  def getAdaptStateTuplePlaceholders(self):
    sxxt_p = tf.placeholder(tf.float32, [None,self._input_dims+1,self._input_dims+1])
    sxy_p = tf.placeholder(tf.float32, [None,self._input_dims+1,self._num_y])
    count_p = tf.placeholder(tf.float32,[None])

    spxx_p = tf.placeholder(tf.float32, [None,self._nprojs])
    spx_p = tf.placeholder(tf.float32, [None,self._nprojs])
    spxy_p = tf.placeholder(tf.float32, [None,self._nprojs, self._num_y])
    sy_p = tf.placeholder(tf.float32, [None,self._num_y])

    self._num_states = 7
    return (sxxt_p,sxy_p,spx_p,spxx_p,sy_p,spxy_p,count_p)


  def getNeededStates(self,combined_state_tuple_current,k,combined_state_tuple_next):
    (sxxt_n,sxy_n,spx_n,spxx_n,sy_n,spxy_n,count_n) = combined_state_tuple_current
    (sxxt_needed,sxy_needed,spx_needed,spxx_needed,sy_needed,spxy_needed,count_needed) = combined_state_tuple_next
    sxxt_needed.append(sxxt_n[k,:,:])
    sxy_needed.append(sxy_n[k,:,:])
    spx_needed.append(spx_n[k,:])
    spxx_needed.append(spxx_n[k,:])
    sy_needed.append(sy_n[k,:])
    spxy_needed.append(spxy_n[k,:,:])
    count_needed.append(count_n[k])
    return (sxxt_needed,sxy_needed,spx_needed,spxx_needed,sy_needed,spxy_needed,count_needed)

  def initialize_zero_states(self,combined_state_tuple_n):
    (sxxt_needed,sxy_needed,spx_needed,spxx_needed,sy_needed,spxy_needed,count_needed) = combined_state_tuple_n
    
    sxxt_needed.append(np.zeros((self._input_dims+1, self._input_dims+1)))
    sxy_needed.append(np.zeros((self._input_dims+1, self._num_y)))
    spx_needed.append(np.zeros((self._nprojs)))
    spxx_needed.append(np.zeros((self._nprojs)))
    sy_needed.append(np.zeros((self._num_y)))
    spxy_needed.append(np.zeros((self._nprojs, self._num_y)))
    count_needed.append(0)
    return (sxxt_needed,sxy_needed,spx_needed,spxx_needed,sy_needed,spxy_needed,count_needed)

  def getZeroStateNumpyArray(self,batch_size_f):
    sxxt_needed = np.zeros((batch_size_f, self._input_dims+1, self._input_dims+1))
    sxy_needed = np.zeros((batch_size_f,self._input_dims+1,self._num_y))
    
    spx_needed = np.zeros((batch_size_f,self._nprojs))
    sy_needed = np.zeros((batch_size_f,self._num_y))
    spxx_needed = np.zeros((batch_size_f,self._nprojs))
    spxy_needed = np.zeros((batch_size_f,self._nprojs, self._num_y))
    
    count_needed = np.zeros((batch_size_f))
    return (sxxt_needed,sxy_needed,spx_needed,spxx_needed,sy_needed,spxy_needed,count_needed)
