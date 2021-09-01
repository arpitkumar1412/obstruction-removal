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
from keras.preprocessing.image import img_to_array, array_to_img
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import tensor
from torch.optim import SGD
from torch.nn import BCELoss, MSELoss
from torchvision import transforms


width=64
height=56
final_width, final_height, final_channels = int(width/1), int(height/1), 3
prod_width, prod_height = 128,768
device=torch.device('cuda')
import os
cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

def convert_pred(vid, shape):
  edit_vid = np.zeros((6,128,128,1), dtype=np.uint8)
  for i in range(vid.shape[0]):
    img = Image.fromarray(vid[i], 'RGB').convert('L')
    img = img.resize((128,128))
    edit_vid[i,:,:,0] = img

  edit_vid = torch.from_numpy(np.reshape(edit_vid, shape))
  return edit_vid.to(device)


def convert_actual(vid):
  edit_vid = np.zeros((1,6,64,64,3), dtype=np.uint8)
  for i in range(vid.shape[0]):
    img = Image.fromarray(vid[i])
    img = img.resize((64,64))
    edit_vid[0,i] = img

  edit_vid = torch.from_numpy(edit_vid)
  return edit_vid.to(device)

def load_img(img):
  img = Image.fromarray((img).astype(np.uint8)).resize((128,128))
  img = transforms.ToTensor()(img)
  return img[None].to(device)

def load_image(img):
    img = torch.from_numpy(img).permute(3,1,2,0).float()
    return img[None].to(device)

def get_flow_ini(vid):
  flow_val = np.zeros((6,64,64,3), dtype=np.uint8)
  for j in range(5):
    image1 = load_img(vid[j])
    image2 = load_img(vid[j+1])
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    flow_low, flow_up = model_flow(image1, image2, iters=4, test_mode=True)

    img = image1[0].permute(1,2,0).cpu().detach().numpy()
    flo = flow_up[0].permute(1,2,0).cpu().detach().numpy()
    flo = flow_viz.flow_to_image(flo)
    flo = Image.fromarray(flo).resize((64,64))
    flow_val[j] = transforms.ToTensor()(flo).permute(1,2,0).float()
    return flow_val

def get_flow(vid):
  flow_val = np.zeros((1,6,64,64,3), dtype=np.uint8)
  for j in range(6):
    image1 = load_img(vid[0,j])
    image2 = load_img(vid[0,j+1])
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    flow_low, flow_up = model_flow(image1, image2, iters=4, test_mode=True)

    img = image1[0].permute(1,2,0).cpu().detach().numpy()
    flo = flow_up[0].permute(1,2,0).cpu().detach().numpy()
    flo = flow_viz.flow_to_image(flo)
    flo = Image.fromarray(flo).resize((64,64))
    flow_val[0,j] = transforms.ToTensor()(flo).permute(1,2,0).float()
    return flow_val

def combine_images(data):
    combined_item = np.zeros((prod_width*6,prod_height,3), dtype=np.uint8)
    frames=6
    for j in range(frames):
        pixels = Image.fromarray(data[j,:,:,:].astype(np.uint8))
        pixels = pixels.resize((prod_height,prod_width))
        pixels = img_to_array(pixels)
        combined_item[prod_width*j:prod_width*(j+1),:,:] = pixels
    return combined_item

back = load_model('../models/back_ref.hdf5')
obs = load_model('../models/obs_ref.hdf5')
print("models loaded")

mixed = np.load('../data/reflection-mixed.npy')
inp = np.load('../data/reflection-inp.npy')
print("data loaded")

#pred_layer1_back = back.predict(tf.expand_dims(inp[i], axis=0))
#pred_layer1_obs = obs.predict(tf.expand_dims(inp[i], axis=0))
#print(pred_layer1_back.shape)
#pred_layer1_back = tf.squeeze(pred_layer1_back)[0]
#array_to_img(pred_layer1_back).save('prediction_layer1_1.png')
# array_to_img(pred_layer1_obs).save('prediction_layer1_2.png')

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
model_flow.to(device)
model_flow.eval()

class conv3d_bn(nn.Module):
  def __init__(self, in_ch, out_ch, k=(1, 1, 1), s=(1, 1, 1), p=(0, 0, 0)):
    super().__init__()
    self.in_ch = in_ch
    self.out_ch = out_ch
    self.k = k
    self.s = s
    self.p = p
    self.conv3d = nn.Sequential(
        nn.Conv3d(self.in_ch,
                  self.out_ch,
                  kernel_size=self.k,
                  stride=self.s,
                  padding=self.p), nn.BatchNorm3d(self.out_ch),
        nn.ReLU(inplace=True))

  def forward(self, x):
    return self.conv3d(x)


class trans3d_bn(nn.Module):
  def __init__(self, in_ch, out_ch=(64, 64), k=(1, 1, 1), s=(1, 1, 1), p=(0, 0, 0)):
    super().__init__()
    self.in_ch = in_ch
    self.out_ch = out_ch
    self.k = k
    self.s = s
    self.p = p
    self.trans3d = nn.Sequential(
        nn.ConvTranspose3d(self.in_ch,
                           self.out_ch[0],
                           kernel_size=self.k,
                           stride=self.s,
                           padding=self.p),
        nn.BatchNorm3d(self.out_ch[0]),
        nn.ReLU(inplace=True),
        conv3d_bn(self.out_ch[0],
                  self.out_ch[1],
                  k=(3, 5, 3),
                  s=(1, 1, 1),
                  p=(1, 2, 1))  # default kernel_size = 2,4,4 or 4,4,4
        )

  def forward(self, x):
    return self.trans3d(x)


class Encoder_Decoder(nn.Module):
  def __init__(self, n_classes=1):
    super().__init__()
    self.n_classes = n_classes

    """
    encoder
    """
    self.encoder = torch.hub.load(
                TORCH_R2PLUS1D,
                model_name,
                num_classes=MODELS[model_name],
                pretrained=False,
            ).to(device)

    """
    5th layer
    """
    self.layer_5 = trans3d_bn(in_ch=512,
                            out_ch=(128,128),
                            k=(2, 5, 4),
                            s=(2, 1, 2),
                            p=(0, 1, 1))  # 64, 16, 14, 14
    self.layer_4 = conv3d_bn(in_ch=256, out_ch=128, k=(2,2,1), p=(1,0,0))

    """
    4th layer
    """
    self.layer_54 = trans3d_bn(in_ch=256,
                              out_ch=(128, 64),
                              k=(3, 3, 4),
                              s=(1, 1, 2),
                              p=(1, 1, 1))  # 64, 32, 28, 28
    self.layer_3 = conv3d_bn(in_ch=128, out_ch=64, s=(1,1,1), p=(0,2,0))

    """
    3rd layer
    """
    self.layer_543 = trans3d_bn(in_ch=128,
                                out_ch=(64, 32),
                                k=(4, 6, 4),
                                s=(1, 2, 2),
                                p=(1, 0, 1))  # 32, 32, 56, 56, default kernel_size = 4
    self.layer_2 = conv3d_bn(in_ch=64, out_ch=32, s=(1, 1, 1), p=(0,18,0))

    """
    2nd layer
    """
    self.layer_5432 = trans3d_bn(in_ch=64,
                                  out_ch=(32, 16),
                                  k=(5, 4, 3),
                                  s=(1, 1, 1),
                                  p=(2, 0, 1))  # 32, 32, 112, 112, default kernel_size = 4
    self.layer_1 = conv3d_bn(in_ch=64, out_ch=16, p=(0,18,0))

    """
    1st layer
    """
    self.final_2 = nn.ConvTranspose3d(64,32,
                                    kernel_size=(1, 3, 3),
                                    stride=(1, 1, 1),
                                    padding=(0, 1, 1))  # 32, 64, 224, 224
    self.final_1 = nn.ConvTranspose3d(32,16,
                                    kernel_size=(1, 3, 3),
                                    stride=(1, 1, 1),
                                    padding=(0, 1, 1))  # 32, 64, 224, 224
    self.final_0 = nn.ConvTranspose3d(16,8,
                                    kernel_size=(1, 3, 3),
                                    stride=(1, 1, 1),
                                    padding=(0, 1, 1))  # 32, 64, 224, 224
    self.out = nn.Conv3d(
        8,7,
        kernel_size=(3, 5, 5),
        stride=(1, 1, 1),
        padding=(1, 2, 2))  # n*3, 64, 224, 224, default kernel_size = 4,4,4


  def forward(self, inputs):
    """
    init
    """
    data = inputs['inp']    #prepare data for entry to encoder
    pred_this = inputs['pred_this']
    pred_that = inputs['pred_that']
    flo_this = inputs['flo_this']

    out1 = self.encoder.stem(data)       #outputs of encoder model at various points
    out2 = self.encoder.layer1(out1)
    out3 = self.encoder.layer2(out2)
    out4 = self.encoder.layer3(out3)
    out5 = self.encoder.layer4(out4)

    out5 = torch.cat([out5, convert_pred(pred_this, (1,512,8,24,1))], 3)    #setting inputs for background decoder
    out4 = torch.cat([out4, convert_pred(pred_that, (1,256,16,24,1))], 3)
    out3 = torch.cat([out3, convert_pred(flo_this, (1,128,32,12,2))], 3)

    inp_5 = out5.permute(0,1,4,3,2)
    inp_4 = out4.permute(0,1,4,3,2)
    inp_3 = out3.permute(0,1,4,3,2)
    inp_2 = out2.permute(0,1,4,3,2)
    inp_1 = out1.permute(0,1,4,3,2)
    """
    layer 5
    """
    layer_5 = self.layer_5(inp_5)
    layer_4 = self.layer_4(inp_4)
    out_54 = torch.cat([layer_5, layer_4], 1)
    """
    layer 4
    """
    layer_54 = self.layer_54(out_54)
    layer_3 = self.layer_3(inp_3)
    out_543 = torch.cat([layer_54, layer_3], 1)
    """
    layer 3
    """
    layer_543 = self.layer_543(out_543)
    layer_2 = self.layer_2(inp_2)
    out_5432 = torch.cat([layer_543, layer_2], 1)

    """
    output layer
    """
    final_2 = self.final_2(out_5432)
    final_1 = self.final_1(final_2)
    final_0 = self.final_0(final_1)
    out = self.out(final_0)
    return out

for i in range(1000):
    decode_back = torch.load('../models_2_n/back-ref.pth')
    decode_obs = torch.load('../models_2_n/obs-ref.pth')

    data = load_image(mixed[i,:6])    #prepare data for entry to encoder

    pred_back = np.asarray(tf.squeeze(back(tf.expand_dims(inp[i], axis=0))), dtype=np.uint8)  #output from pre-trained model
    pred_obs = np.asarray(tf.squeeze(obs(tf.expand_dims(inp[i], axis=0))), dtype=np.uint8)
    flo_back = get_flow_ini(pred_back)
    flo_obs = get_flow_ini(pred_obs)

    layers=6
    for l in range(layers):
        inputs_back = {'inp': data,
          'pred_this': pred_back,
          'pred_that': pred_obs,
          'flo_this': flo_back
        }
        inputs_obs = {'inp': data,
          'pred_this': pred_obs,
          'pred_that': pred_back,
          'flo_this': flo_obs
        }

        pred_back = decode_back(inputs_back)
        pred_obs = decode_obs(inputs_obs)

        flo_back = np.squeeze(get_flow(pred_back.permute(0,1,4,3,2).cpu().detach().numpy()))
        flo_obs = np.squeeze(get_flow(pred_obs.permute(0,1,4,3,2).cpu().detach().numpy()))

        pred_back = pred_back[:,:6]
        pred_obs = pred_obs[:,:6]
        if l!=layers-1:
            pred_back = np.squeeze(pred_back.cpu().detach().numpy())
            pred_obs = np.squeeze(pred_obs.cpu().detach().numpy())

    yhat_back = pred_back.permute(0,1,4,3,2)
    yhat_obs = pred_obs.permute(0,1,4,3,2)
        #
        # pred_back = np.squeeze(pred_back.detach().numpy())
        # pred_obs = np.squeeze(pred_obs.detach().numpy())

    #predict results
    # model = load_model('../models_3/model_ref_back.h5', compile=False)
    pred_back = np.squeeze(yhat_back.cpu().detach().numpy())
    pred_obs = np.squeeze(yhat_obs.cpu().detach().numpy())
    print(pred_back.shape)
    print(pred_obs.shape)
    pred_back = combine_images(pred_back)
    pred_obs = combine_images(pred_obs)
    img_1 = Image.fromarray(pred_back)
    img_2 = Image.fromarray(pred_obs)
    print(i)
     # pixels = img_to_array(pixels)
# pixels = (pred_back - 127.5) / 127.5
# img = expand_dims(pixels, 0)
# print(img.size)
# result = model.predict(img)
# result = (result + 1) / 2.0
# print(result[0].shape)
# result = Image.fromarray(result[0].astype(np.uint8))
# result.save('prediction.png')
