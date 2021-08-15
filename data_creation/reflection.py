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

mixed = np.zeros((batch,7,width,height,3), dtype=np.uint8)
vid1 = np.zeros((batch,6,final_width, final_height,3), dtype=np.uint8)
vid2 = np.zeros((batch,6,final_width, final_height,3), dtype=np.uint8)

k = 0
loc = '../../data/reflection-final/'

#convert image files to .npy files for the model to process
for j in os.listdir(loc):
  mixed_path = loc+str(j)+'/mixed/'
  vid1_path = loc+str(j)+'/vid1/'
  vid2_path = loc+str(j)+'/vid2/'

  i=0
  for mix in os.listdir(mixed_path):
    mixed[k,i,:,:] = np.asarray(Image.open(os.path.join(mixed_path, mix)).resize((height,width)), dtype=np.uint8)
    i+=1

  i=0
  for back in os.listdir(vid1_path):
    vid1[k,i,:,:] = np.asarray(Image.open(os.path.join(vid1_path, back)).resize((final_height,final_width)), dtype=np.uint8)
    i+=1
    if i==6:
      break

  i=0
  for obs in os.listdir(vid2_path):
    vid2[k,i,:,:,:] = np.asarray(Image.open(os.path.join(vid2_path, obs)).resize((final_height,final_width)), dtype=np.uint8)
    i+=1
    if i==6:
      break
  k+=1

np.save('D:/reflection-mixed.npy', mixed)
np.save('D:/reflection-vid1.npy', vid1)
np.save('D:/reflection-vid2.npy', vid2)
