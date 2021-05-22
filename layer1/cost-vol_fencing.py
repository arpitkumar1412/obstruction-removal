import numpy as np
import tensorflow as tf
import scipy
from scipy import signal

mixed = np.load('../../data/reflection-mixed.npy')
print(mixed.shape)
print("loading inputs")

width=64
height=56

def cal_cost(features):
  print("calculating cost")
  cost = np.zeros((1000, 6, width, height, 3))
  for i in range(features.shape[0]):
    for j in range(6):
      flat_feature1 = mixed[i,j,:,:,0]
      flat_feature2 = mixed[i,j+1,:,:,0]
      cost_vol = scipy.signal.correlate2d(flat_feature1, flat_feature2, mode='same')   #normalize here ############################
      img = np.reshape(cost_vol, (width, height))
      cost[i,j,:,:,0] = img

  inp = np.zeros((1000, 6, 3*width, height, 3))
  for i in range(1000):
    for j in range(6):
      inp[i][j] = tf.concat([mixed[i][j], mixed[i][j+1], cost[i][j]], axis=0)
  return tf.convert_to_tensor(inp)
inp = cal_cost(mixed)
