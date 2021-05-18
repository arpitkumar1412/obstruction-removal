width=64
height=56
final_width, final_height, final_channels = int(width/1), int(height/1), 3
batch=1000
import scipy
from scipy import signal
import os
from PIL import Image
import PIL.ImageOps
import numpy as np
import tensorflow as tf

# mixed = np.zeros((batch,7,width,height,3), dtype=np.uint8)
# vid1 = np.zeros((batch,6,final_width, final_height,3), dtype=np.uint8)
vid2 = np.zeros((batch,6,final_width, final_height,3), dtype=np.uint8)

k = 0
loc = 'D:/reflection-final/'
for j in os.listdir(loc):
  mixed_path = loc+str(j)+'/mixed/'
  vid1_path = loc+str(j)+'/vid1/'
  vid2_path = loc+str(j)+'/vid2/'
  print(mixed_path)
  # i=0
  # for mix in os.listdir(mixed_path):
  #   mixed[k,i,:,:] = np.asarray(Image.open(os.path.join(mixed_path, mix)).resize((height,width)), dtype=np.uint8)
  #   i+=1
  # i=0
  # for back in os.listdir(vid1_path):
  #   vid1[k,i,:,:] = np.asarray(Image.open(os.path.join(vid1_path, back)).resize((final_height,final_width)), dtype=np.uint8)
  #   i+=1
  #   if i==6:
  #     break;
  i=0
  for obs in os.listdir(vid2_path):
    vid2[k,i,:,:,:] = np.asarray(Image.open(os.path.join(vid2_path, obs)).resize((final_height,final_width)), dtype=np.uint8)
    i+=1
    if i==6:
      break;
  k+=1

# def cal_cost(features):
#   print("calculating cost")
#   cost = np.zeros((batch, 6, width, height, 3), dtype=np.uint8)
#   for i in range(features.shape[0]):
#     for j in range(6):
#       flat_feature1 = mixed[i,j,:,:,0]
#       flat_feature2 = mixed[i,j+1,:,:,0]
#       cost_vol = scipy.signal.correlate2d(flat_feature1, flat_feature2, mode='same')   #normalize here ############################
#       img = np.reshape(cost_vol, (width, height))
#       cost[i,j,:,:,0] = img
#
#       flat_feature1 = mixed[i,j,:,:,1]
#       flat_feature2 = mixed[i,j+1,:,:,1]
#       cost_vol = scipy.signal.correlate2d(flat_feature1, flat_feature2, mode='same')   #normalize here ############################
#       img = np.reshape(cost_vol, (width, height))
#       cost[i,j,:,:,0] = img
#
#       flat_feature1 = mixed[i,j,:,:,2]
#       flat_feature2 = mixed[i,j+1,:,:,2]
#       cost_vol = scipy.signal.correlate2d(flat_feature1, flat_feature2, mode='same')   #normalize here ############################
#       img = np.reshape(cost_vol, (width, height))
#       cost[i,j,:,:,0] = img
#     print(i)
#   inp = np.zeros((batch, 6, 3*width, height, 3), dtype=np.uint8)
#   for i in range(batch):
#     for j in range(6):
#       inp[i][j] = tf.concat([mixed[i][j], mixed[i][j+1], cost[i][j]], axis=0)
#   return tf.convert_to_tensor(inp)
# inp = cal_cost(mixed)
# np.save('D:/reflection-mixed.npy', mixed)
# np.save('D:/reflection-vid1.npy', vid1)
np.save('D:/reflection-vid2.npy', vid2)
# np.save('D:/reflection-inp.npy', inp)
