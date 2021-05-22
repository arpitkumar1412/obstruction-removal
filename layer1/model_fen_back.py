width=64
height=56
final_width, final_height, final_channels = int(width/1), int(height/1), 3
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
import scipy
from scipy import signal
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

inp = np.load('../../data/fencing-inp.npy')
vid1 = np.load('../../data/fencing-vid1.npy')
vid2 = np.load('../../data/fencing-vid2.npy')
print(inp.shape)
print("loading inputs")

def feature_extractor_and_layer_flow_estimator():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Conv2D(16, (3, 3), (2, 2), 'same', name='a', input_shape=(6,3*width,height,3), data_format='channels_last'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Conv2D(16, (3, 3), (1, 1), 'same', name='b'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Conv2D(32, (3, 3), (2, 2), 'same', name='c'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), 'same', name='d'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Conv2D(64, (3, 3), (2, 2), 'same', name='e'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Conv2D(64, (3, 3), (1, 1), 'same', name='f'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Conv2D(128, (3, 3), (2, 2), 'same', name='g'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Conv2D(128, (3, 3), (1, 1), 'same', name='h'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Conv2D(64, (3, 3), (1, 1), 'same', name='i'))
  model.add(tf.keras.layers.LeakyReLU(0.2))
  model.add(tf.keras.layers.Conv2D(64, (3, 3), (1, 1), 'same', name='j'))
  model.add(tf.keras.layers.LeakyReLU(0.2))
  model.add(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), 'same', name='k'))
  model.add(tf.keras.layers.LeakyReLU(0.2))
  model.add(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), 'same', name='l'))
  model.add(tf.keras.layers.LeakyReLU(0.2))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(6*final_width*final_height*final_channels, name='m'))
  model.add(tf.keras.layers.Reshape((6, final_width, final_height, final_channels)))
  return model

# def cal_cost(features):
#   print("calculating cost")
#   cost = np.zeros((1000, 6, width, height, 3))
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
#   inp = np.zeros((1000, 6, 3*width, height, 3))
#   for i in range(1000):
#     for j in range(6):
#       inp[i][j] = tf.concat([mixed[i][j], mixed[i][j+1], cost[i][j]], axis=0)
#   return tf.convert_to_tensor(inp)
# inp = cal_cost(mixed)
# np.save('reflection-inp.npy', inp)
back = feature_extractor_and_layer_flow_estimator()
back.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
X_train, X_test, y_train, y_test = train_test_split(inp, vid1, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
checkpoint = ModelCheckpoint("../../models/back_fen.hdf5", monitor='loss', verbose=1,save_best_only=True, mode='auto', period=10)
callbacks = [checkpoint]
back.fit(X_train, y_train, batch_size=50, epochs=150, validation_data=(X_test, y_test), shuffle=True, callbacks=callbacks)
