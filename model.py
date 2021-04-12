width=64
height=56
final_width, final_height, final_channels = int(width/2), int(height/2), 3
import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
import tensorflow as tf
from sklearn.preprocessing import normalize
import scipy
from scipy import signal
import os
from PIL import Image
from skimage import measure

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

model1 = feature_extractor_and_layer_flow_estimator()
model1.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
X = tf.convert_to_tensor(inp)
y = tf.convert_to_tensor(vid1)
print(X.shape, model1.input)
print(y.shape, model1.output)
model1.fit(X, y, batch_size=50, epochs=10000)
