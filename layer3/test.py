from PIL import Image
import numpy as np
from numpy import asarray

file = '../layer2_prediction/back_ref/back_ref0.png'
img = Image.open(file).resize((256,256*6))
src = list()
src.append(asarray(img))
src = asarray(src)
Image.fromarray(src[0]).save('test.png')
