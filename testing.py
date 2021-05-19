import numpy as np
from numpy import expand_dims
import os
from pathlib import Path
import time
import warnings
from sklearn.metrics import accuracy_score
import sys
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import torch
import torchvision
import torch.nn as nn
from torch import tensor
from torch.optim import SGD
from torch.nn import BCELoss, MSELoss
from torchvision import transforms


width=64
height=56
final_width, final_height, final_channels = int(width/1), int(height/1), 3
prod_width, prod_height = 128,768
i=1
DEVICE='cuda'

def convert_pred(vid, shape):
  edit_vid = np.zeros((6,128,128,1), dtype=np.uint8)
  for i in range(vid.shape[0]):
    img = Image.fromarray(vid[i,:,:,:], 'RGB').convert('L')
    img = img.resize((128,128))
    edit_vid[i,:,:,0] = img

  edit_vid = torch.from_numpy(np.reshape(edit_vid, shape))
  return edit_vid

def convert_actual(vid):
  edit_vid = np.zeros((1,6,64,64,3), dtype=np.uint8)
  for i in range(vid.shape[0]):
    img = Image.fromarray(vid[i,:,:,:])
    img = img.resize((64,64))
    edit_vid[0,i,:,:,:] = img

  edit_vid = torch.from_numpy(edit_vid)
  return edit_vid

def load_img(img):
  img = Image.fromarray((img).astype(np.uint8)).resize((128,128))
  img = transforms.ToTensor()(img)
  return img[None].to(DEVICE)

def load_image(img):
    img = torch.from_numpy(img).permute(3,1,2,0).float()
    return img[None]

def get_flow_ini(vid):
  flow_val = np.zeros((6,64,64,3), dtype=np.uint8)
  for j in range(5):
    image1 = load_img(vid[j,:,:,:])
    image2 = load_img(vid[j+1,:,:,:])
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    flow_low, flow_up = model_flow(image1, image2, iters=4, test_mode=True)

    img = image1[0].permute(1,2,0).cpu().detach().numpy()
    flo = flow_up[0].permute(1,2,0).cpu().detach().numpy()
    flo = flow_viz.flow_to_image(flo)
    flo = Image.fromarray(flo).resize((64,64))
    flow_val[j,:,:,:] = transforms.ToTensor()(flo).permute(1,2,0).float()
    return flow_val

def get_flow(vid):
  flow_val = np.zeros((1,6,64,64,3), dtype=np.uint8)
  for j in range(6):
    image1 = load_img(vid[0,j,:,:,:])
    image2 = load_img(vid[0,j+1,:,:,:])
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    flow_low, flow_up = model_flow(image1, image2, iters=4, test_mode=True)

    img = image1[0].permute(1,2,0).cpu().detach().numpy()
    flo = flow_up[0].permute(1,2,0).cpu().detach().numpy()
    flo = flow_viz.flow_to_image(flo)
    flo = Image.fromarray(flo).resize((64,64))
    flow_val[0,j,:,:,:] = transforms.ToTensor()(flo).permute(1,2,0).float()
    return flow_val

back = load_model('../models/back_ref.hdf5')
obs = load_model('../models/obs_ref.hdf5')
print("models loaded")

mixed = np.load('../data/mixed.npy')[1000:2000,:,:,:,:]
inp = np.load('../data/inp.npy')[1000:2000,:,:,:,:]
print("data loaded")

TORCH_R2PLUS1D = "moabitcoin/ig65m-pytorch"  # From https://github.com/moabitcoin/ig65m-pytorch
MODELS = {
    # Model name followed by the number of output classes.
    "r2plus1d_34_32_ig65m": 359,
    "r2plus1d_34_32_kinetics": 400,
    "r2plus1d_34_8_ig65m": 487,
    "r2plus1d_34_8_kinetics": 400,
}
model_name = 'r2plus1d_34_8_kinetics'

model_encoder = torch.hub.load(
            TORCH_R2PLUS1D,
            model_name,
            num_classes=MODELS[model_name],
            pretrained=True,
        )
print("encoder model created")

sys.path.append('../RAFT/core/')
from raft import RAFT
from argparse import Namespace
from utils import flow_viz
from utils.utils import InputPadder
args = Namespace(alternate_corr=False, mixed_precision=False, model='../RAFT/raft-things.pth', small=False)
model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(args.model))
print("model-optical flow created")
model_flow = model.module
model_flow.to(DEVICE)
model_flow.eval()

decode_back = torch.load('../models_2/back-ref.pth')
decode_obs = torch.load('../models_2/obs-ref.pth')

# compute the model output
data = load_image(mixed[i,:6,:,:,:])    #prepare data for entry to encoder
out1 = model_encoder.stem(data)        #outputs of encoder model at various points
out2 = model_encoder.layer1(out1)
out3 = model_encoder.layer2(out2)
out4 = model_encoder.layer3(out3)
out5 = model_encoder.layer4(out4)

print("calculating output for pretrained model")
pred_back = np.asarray(tf.squeeze(back(tf.expand_dims(inp[i], axis=0))), dtype=np.uint8)  #output from pre-trained model
pred_obs = np.asarray(tf.squeeze(obs(tf.expand_dims(inp[i], axis=0))), dtype=np.uint8)
flo_back = get_flow_ini(pred_back)
flo_obs = get_flow_ini(pred_obs)

for l in range(layers):
    # print("layer: "+str(l))
    out5_b = torch.cat([out5, convert_pred(pred_back, (1,512,8,24,1))], 3)    #setting inputs for background decoder
    out4_b = torch.cat([out4, convert_pred(pred_obs, (1,256,16,24,1))], 3)
    out3_b = torch.cat([out3, convert_pred(flo_back, (1,128,32,12,2))], 3)

    out5_o = torch.cat([out5, convert_pred(pred_obs, (1,512,8,24,1))], 3)     #setting inputs for obstruction decoder
    out4_o = torch.cat([out4, convert_pred(pred_back, (1,256,16,24,1))], 3)
    out3_o = torch.cat([out3, convert_pred(flo_obs, (1,128,32,12,2))], 3)
    inputs_back = {'inp_5' : out5_b.permute(0,1,4,3,2),
              'inp_4' : out4_b.permute(0,1,4,3,2),
              'inp_3' : out3_b.permute(0,1,4,3,2),
              'inp_2' : out2.permute(0,1,4,3,2),
              'inp_1' : out1.permute(0,1,4,3,2)
            }
    inputs_obs = {'inp_5' : out5_o.permute(0,1,4,3,2),
              'inp_4' : out4_o.permute(0,1,4,3,2),
              'inp_3' : out3_o.permute(0,1,4,3,2),
              'inp_2' : out2.permute(0,1,4,3,2),
              'inp_1' : out1.permute(0,1,4,3,2)
            }

    pred_back = decode_back(inputs_back)
    pred_obs = decode_obs(inputs_obs)

    flo_back = np.squeeze(get_flow(pred_back.permute(0,1,4,3,2).cpu().detach().numpy()))
    flo_obs = np.squeeze(get_flow(pred_obs.permute(0,1,4,3,2).cpu().detach().numpy()))

    pred_back = pred_back[:,:6,:,:,:]
    pred_obs = pred_obs[:,:6,:,:,:]
    yhat_back = pred_back.permute(0,1,4,3,2)
    yhat_obs = pred_obs.permute(0,1,4,3,2)

    pred_back = np.squeeze(pred_back.detach().numpy())
    pred_obs = np.squeeze(pred_obs.detach().numpy())

#predict results
model = load_model('../models_3/model_ref_back.h5')
img = Image.fromarray(pred_back.astype(np.uint8))
pixels = img_to_array(pixels)
pixels = (pixels - 127.5) / 127.5
img = expand_dims(pixels, 0)
print(img.size)
result = model.predict(img)
result = (result + 1) / 2.0
print(result[0].shape)
result = Image.fromarray(result[0].astype(np.uint8))
result.save('prediction.png')
