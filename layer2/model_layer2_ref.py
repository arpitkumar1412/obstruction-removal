import numpy as np
import os
from pathlib import Path
import time
import warnings
from sklearn.metrics import accuracy_score
import sys
import torch
import torchvision
import torch.nn as nn
from PIL import Image
import tensorflow as tf
from torch import tensor
from keras.models import load_model
from torch.optim import SGD
from torch.nn import BCELoss, MSELoss
from torchvision import transforms


width=64
height=56
final_width, final_height, final_channels = int(width/1), int(height/1), 3
batch=1000
DEVICE='cuda'

def convert_pred(vid, shape):
  edit_vid = np.zeros((6,128,128,1), dtype=np.uint8)
  for i in range(vid.shape[0]):
    img = Image.fromarray(vid[i], 'RGB').convert('L')
    img = img.resize((128,128))
    edit_vid[i,:,:,0] = img

  edit_vid = torch.from_numpy(np.reshape(edit_vid, shape))
  return edit_vid

def convert_actual(vid):
  edit_vid = np.zeros((1,6,64,64,3), dtype=np.uint8)
  for i in range(vid.shape[0]):
    img = Image.fromarray(vid[i])
    img = img.resize((64,64))
    edit_vid[0,i] = img

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

back = load_model('../../models/back_ref.hdf5')
obs = load_model('../../models/obs_ref.hdf5')
print("models loaded")

inp = np.load('../../data/reflection-inp.npy')
vid1 = np.load('../../data/reflection-vid1.npy')
vid2 = np.load('../../data/reflection-vid2.npy')
mixed = np.load('../../data/reflection-mixed.npy')
print(inp.shape)
print("loading inputs")

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

sys.path.append('../../RAFT/core/')
from raft import RAFT
from argparse import Namespace
from utils import flow_viz
from utils.utils import InputPadder
args = Namespace(alternate_corr=False, mixed_precision=False, model='../../RAFT/raft-things.pth', small=False)
model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(args.model))
print("model-optical flow created")
model_flow = model.module
model_flow.to(DEVICE)
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


class Decoder(nn.Module):
  def __init__(self, n_classes=1):
    super().__init__()
    self.n_classes = n_classes

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
    inp_5 = inputs['inp_5']
    inp_4 = inputs['inp_4']
    inp_3 = inputs['inp_3']
    inp_2 = inputs['inp_2']
    inp_1 = inputs['inp_1']
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


# train the model   background
def train_model(layers, epochs):
  # define the optimization
  # decode_back = Decoder(1)    #defining the model
  # decode_obs = Decoder(1)
  decode_back = torch.load('../../models_2/back-ref.pth')
  decode_obs = torch.load('../../models_2/obs-ref.pth')
  criterion = MSELoss()
  optimizer_back = SGD(decode_back.parameters(), lr=0.01, momentum=0.9)
  optimizer_obs = SGD(decode_obs.parameters(), lr=0.01, momentum=0.9)

  # enumerate epochs
  for epoch in range(413,epochs):
    running_loss_back = 0
    running_loss_obs = 0
    # enumerate mini batches
    batch = 0
    for i in range(batch,batch+100):
    # clear the gradients
        batch = (batch+100)%1000
        optimizer_back.zero_grad()
        optimizer_obs.zero_grad()
        # compute the model output
        data = load_image(mixed[i,:6])    #prepare data for entry to encoder
        out1 = model_encoder.stem(data)        #outputs of encoder model at various points
        out2 = model_encoder.layer1(out1)
        out3 = model_encoder.layer2(out2)
        out4 = model_encoder.layer3(out3)
        out5 = model_encoder.layer4(out4)

        pred_back = np.asarray(tf.squeeze(back(tf.expand_dims(inp[i], axis=0))), dtype=np.uint8)  #output from pre-trained model
        pred_obs = np.asarray(tf.squeeze(obs(tf.expand_dims(inp[i], axis=0))), dtype=np.uint8)
        flo_back = get_flow_ini(pred_back)
        flo_obs = get_flow_ini(pred_obs)

        flo_back_act = np.squeeze(get_flow(convert_actual(vid1[i]).float().detach().numpy()))
        flo_obs_act = np.squeeze(get_flow(convert_actual(vid2[i]).float().detach().numpy()))

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

            pred_back = pred_back[:,:6]
            pred_obs = pred_obs[:,:6]
            if l!=layers-1:
                pred_back = np.squeeze(pred_back.detach().numpy())
                pred_obs = np.squeeze(pred_obs.detach().numpy())

        yhat_back = pred_back.permute(0,1,4,3,2)
        yhat_obs = pred_obs.permute(0,1,4,3,2)

        # calculate loss
        loss_back = criterion(yhat_back, convert_actual(vid1[i]).float())+criterion(torch.from_numpy(flo_back).float(),torch.from_numpy(flo_back_act).float())
        loss_obs = criterion(yhat_obs, convert_actual(vid2[i]).float())+criterion(torch.from_numpy(flo_obs).float(),torch.from_numpy(flo_obs_act).float())
        # credit assignment
        loss_back.backward(retain_graph=True)
        loss_obs.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(decode_back.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decode_obs.parameters(), 1.0)
        running_loss_back += loss_back.item()
        running_loss_obs += loss_obs.item()
        # update model weights
        optimizer_back.step()
        optimizer_obs.step()

        print("ref, epoch - "+str(epoch)+", batch - "+str(i)+", running loss background - "+str(running_loss_back)+", running loss obstruction - "+str(running_loss_obs))

    if(epoch%5==0):
        print('saving reflection model, epoch-'+str(epoch))
        torch.save(decode_back, '../../models_2/back-ref.pth')
        torch.save(decode_obs, '../../models_2/obs-ref.pth')
train_model(6,20000)
