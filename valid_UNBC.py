"""valid_UNBC.py
Script to load valid subjects that can be presented in a paper and run the mod_gan test script.
Output will be the results of the subjects on a previous saved and trained model saved on disk.
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pain_gan
import scipy
import numpy as np

from PIL import Image

imagewidth = 28
imageheight = 28
imagechannels = 1

valid_subjects = ['095-tv095', '047-jl047', '109-ib109', '064-ak064']

image_Dataframe = pain_gan.load_imgs()

image_Dataframe = image_Dataframe.loc[image_Dataframe['Subject'].isin(valid_subjects)]

print(image_Dataframe)

#Last usable model was on May 2 2019
model_dir = "5_4_2019_18_1model"

gan = pain_gan.GAN(imagewidth, imageheight, imagechannels)

gan.load_model(model_dir, "AEPHASE_")

X = [scipy.ndimage.imread(f, flatten=True) for f in list(image_Dataframe.index)]
Y = image_Dataframe['Pain'].values

for i in range (len(X)) :
    tempImage = Image.fromarray(X[i])
    X[i] = np.array(tempImage.resize(size=(imageheight,imagewidth)))

X = np.array(X)

X = (X.astype(np.float32) - 127.5) / 127.5
X = np.expand_dims(X, axis=3)

gan.test(X, Y)

exit()

