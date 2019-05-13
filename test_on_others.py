"""test_on_others.py
Script to load images from the IMG_LOAD folder and test the pain representation on the images fed to it

"""
import os
from os import listdir
from os.path import isfile, join
import pain_gan
import scipy
import numpy as np
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

imagewidth=28
imageheight=28
imagechannels=1

#Edit this parameter to match a previous trained keras model directory
model_dir = "5_4_2019_18_1model"

loadable_images = [f for f in listdir('IMG_LOAD') if isfile(join('IMG_LOAD',f))]

X = [scipy.ndimage.imread(join('IMG_LOAD',f), flatten=True) for f in loadable_images]

Y = np.random.randint(0, 15, size=len(X))

for i in range (len(X)):
    tempImage = Image.fromarray(X[i])
    X[i] = np.array(tempImage.resize(size=(imageheight, imagewidth)))

X = np.array(X)

X = (X.astype(np.float32) - 127.5) / 127.5
X = np.expand_dims(X, axis=3)

print(Y)

gan = pain_gan.GAN(imagewidth, imageheight, imagechannels)

gan.load_model(model_dir, extraName="AEPHASE_")

gan.test(X,Y)

exit()