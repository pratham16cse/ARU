import tensorflow as tf
import numpy as np
import ARU as aru


batch_size = 2
seq_len = 100
numF=3
decode_window = 5
adapt = True
num_proj = 4
num_y = 1
full_linear=False

def get_batches():
  w = np.random.uniform(-3.0, 3.0, size=[batch_size, numF])
  w = np.pad(w, [(0, 0), (0, 1)], mode='constant')
  b = np.random.uniform(-1, 1, size=[batch_size])
  x = np.random.uniform(-1.0, 1.0, size=[batch_size,seq_len,numF])
  y = np.zeros([batch_size, seq_len])
  yprev = np.zeros([batch_size,1])
  for t in range(seq_len):
    xpy = np.concatenate((x[:,t,:],yprev),axis=1)
    # print(xpy)
    y[:,t] = np.sum(xpy*w, axis=1) + b + np.random.randn(batch_size)*0.1
    yprev = y[:,t,None]
  return x, y, w, b

total_loss = 0

# print(get_batches(w,b))

x = tf.placeholder(tf.float32, shape=(batch_size, seq_len, numF))
y =  labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, seq_len))

ARU_cell = aru.ARU(numF, num_y, num_proj, full_linear)
xEnc, xDec = tf.split(x, [seq_len-decode_window, decode_window], 1)
yEnc, yDec = tf.split(y, [-1, decode_window], 1)

print "yEnc",yEnc
print "xEnc",np.shape(xEnc)
print "yDec",np.shape(yDec)
print "xDec",np.shape(xDec)

aru_state = ARU_cell.adapt(None, xEnc, tf.expand_dims(yEnc,2))
h, _ = ARU_cell.predict(aru_state, xDec)
# h = tf.reshape(h, [batch_size, decode_window, num_proj+1])
W = tf.get_variable("W", [num_proj+int(full_linear)])
#ypred = tf.einsum('btf,f->bt', h, W) + tf.get_variable("b", [1]) #
ypred = tf.squeeze(tf.layers.dense(inputs = h, units = 1),2)
local_params = ARU_cell.get_Wb(aru_state)
loss = tf.abs(yDec - ypred)
optimiser = tf.train.AdamOptimizer(learning_rate=0.001).minimize(tf.reduce_mean(loss))
session = tf.Session()
session.run(tf.global_variables_initializer())
for i in range(1):
  batch_x, batch_y, w, b = get_batches()
  # LW, LB, Lxxt, Lxy = session.run([local_params[0], local_params[1], local_params[5], local_params[6]],feed_dict={x: batch_x, y: batch_y})
  opt, l, LW, LB, LC = session.run([optimiser,loss, local_params[2], local_params[2], local_params[2]], feed_dict={x: batch_x, y: batch_y})
  print("True W ",  LW, w)
  print("True b ",  LB, b)
  # print("sxxt", Lxxt)
  # print("sxy", Lxy)
  total_loss += np.sum(l, axis=0)
  print i, (total_loss/(i+1)/batch_size/seq_len/decode_window)
