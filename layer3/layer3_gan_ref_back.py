# Pix2pix GAN architecture used here for background layer of reflection problem
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from os import listdir
import numpy as np
from numpy import asarray
from numpy import vstack
from numpy import load
from numpy import expand_dims
from numpy import savez_compressed
import torch
from PIL import Image

# load, split and scale the maps dataset ready for training
width = 768
height = 128
width_final = 56
height_final = 64
batch = 1000
frames = 6

def load_image():

	src_list, tar_list = list(), list()
	path = '../layer2_prediction/back_ref'
	for image in os.listdir(path):
		img = Image.open(path + image)
		np_img = np.array(img)
		index = 0
		for i in range(frames):
			img_curr = np_img[:,index*128:(index+1)*128-1,3]
			img_curr = Image.fromarray(img_curr)
			img_curr = img_curr.resize((width_final, height_final))
			np_img = np.array(img_curr)
			src_list.append(np_img)


	dataset = load('../../data/reflection-vid1.npy')
	for i in range(batch):
		for j in range(frames):
			# load and resize the image
			pixels = dataset[i,j,:,:,:]
			tar_list.append(pixels)

	return src_list, tar_list

# load dataset
[src_images, tar_images] = load_images()
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = '../../data/maps_ref_back.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)

# #load output of 2nd layer as input
# def load_images(size=(width, height)):
#     # load image data
#     dataset = load('../../data/maps_ref_back.npz')
#     src_list, tar_list = list(), list()
#     pred_ref_back = dataset['arr_1']
#     pred_ref_back = asarray(torch.from_numpy(pred_ref_back).permute(0,1,4,3,2))
#     print('Loaded', pred_ref_back.shape)
#
#     for i in range(batch):
#         src_item = np.zeros((width*6,height,3), dtype=np.uint8)
#         k=0
#         for j in range(frames):
#             # load and resize the image
#             pixels = Image.fromarray(pred_ref_back[i,j,:,:,:].astype(np.uint8))
#             pixels = pixels.resize((height,width))
#             # convert to numpy array
#             pixels = img_to_array(pixels)
#             src_item[width*k:width*(k+1),:,:] = pixels
#             k+=1
#         src_list.append(src_item)
#
#     dataset = load('../../data/vid1.npy')
#     dataset = dataset[1000:2000,:,:,:,:]
#     print('Loaded', dataset.shape)
#     # get all the images from data/vid1.npy
#     for i in range(batch):
#         tar_item = np.zeros((width*6,height,3), dtype=np.uint8)
#         k=0
#         for j in range(frames):
#             # load and resize the image
#             pixels = Image.fromarray(dataset[i,j,:,:,:])
#             pixels = pixels.resize((height,width))
#             # convert to numpy array
#             pixels = img_to_array(pixels)
#             tar_item[width*k:width*(k+1),:,:] = pixels
#             k+=1
#         tar_list.append(tar_item)
#     return [asarray(src_list), asarray(tar_list)]



# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # add upsampling layer
  g = Conv2DTranspose(n_filters, kernel_size=(2,2), strides=(2,2), padding='valid', kernel_initializer=init)(layer_in)
  # add batch normalization
  g = BatchNormalization()(g, training=True)
  # conditionally add dropout
  if dropout:
    g = Dropout(0.5)(g, training=True)
  # merge with skip connection
  g = Concatenate()([g, skip_in])
  # relu activation
  g = Activation('relu')(g)
  return g

# define the standalone generator model
def define_generator(image_shape):
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # image input
  in_image = Input(shape=image_shape)
  # encoder model
  e1 = define_encoder_block(in_image, 64, batchnorm=False)
  e2 = define_encoder_block(e1, 128)
  e3 = define_encoder_block(e2, 256)
  e4 = define_encoder_block(e3, 512)
  e5 = define_encoder_block(e4, 512)
  e6 = define_encoder_block(e5, 512)
  e7 = define_encoder_block(e6, 512)
  # bottleneck, no batch norm and relu
  b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
  b = Activation('relu')(b)
  # decoder model
  d1 = decoder_block(b, e7, 512)
  d2 = decoder_block(d1, e6, 512)
  d3 = decoder_block(d2, e5, 512)
  d4 = decoder_block(d3, e4, 512, dropout=False)
  d5 = decoder_block(d4, e3, 256, dropout=False)
  d6 = decoder_block(d5, e2, 128, dropout=False)
  d7 = decoder_block(d6, e1, 64, dropout=False)
  # output
  g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
  out_image = Activation('tanh')(g)
  # define model
  model = Model(in_image, out_image)
  return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
	# save the generator model
	filename2 = '../../models_3/model_ref_back.h5'
	g_model.save(filename2)
	print('>Saved: %s ' % (filename2))

# train pix2pix model
def train(d_model, g_model, gan_model, dataset, n_epochs=10000, n_batch=10):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('ref_back>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % (10) == 0:
			summarize_performance(i, g_model, dataset)

# load image data
dataset = load_real_samples('../../maps_ref_back.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# image_shape = (256,256,3)
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)

# load an image
def load_image(filename, size=(width*6,height)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

# #predict results
# model = load_model('../../models_3/model_ref_back.h5')
# # print(dataset[0][0].shape)
# filename1 = 'test_input_ref_back.png'
# img = Image.fromarray(dataset[0][0].astype(np.uint8))
# img.save(filename1)
# img = load_image('test_input_ref_back.png')
# print(img.size)
# result = model.predict(img)
# # scale from [-1,1] to [0,1]
# result = (result + 1) / 2.0
# print(result[0].shape)
# result = Image.fromarray(result[0].astype(np.uint8))
# result.save('prediction.png')
