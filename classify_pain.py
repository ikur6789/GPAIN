"""classify_pain.py
Script to try and classify what an given input image's pain score is.
Will be using the adversarial autoencoder defined in pain_gan
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pain_gan
import scipy
import sklearn.metrics
import numpy as np
import sys
from PIL import Image

imagewidth = 28
imageheight = 28
imagechannels = 1

MAX_PAIN = 16

image_Dataframe = pain_gan.load_imgs()

model_dir = "5_4_2019_18_1model"

gan = pain_gan.GAN(imagewidth, imageheight, imagechannels)

gan.load_model(model_dir, "AEPHASE_")

X = [scipy.ndimage.imread(f, flatten=True) for f in list(image_Dataframe.index)]
Y = image_Dataframe['Pain'].values

for i in range (len(X)):
    tempImage = Image.fromarray(X[i])
    X[i] = np.array(tempImage.resize(size=(imageheight, imagewidth)))

X = np.array(X)

X = (X.astype(np.float32) - 127.5) / 127.5
X = np.expand_dims(X, axis=3)

#https://stackoverflow.com/questions/6667201/how-to-define-a-two-dimensional-array-in-python
confusion = [[0.0 for x in range(MAX_PAIN)] for y in range(MAX_PAIN)]
TP = 0

for index in range(0, len(X)):
    if index%100 == 0:
        print("Calculating at: %d" % index)
    
    #if index%500 == 0:
        #print(confusion)

    ground_image = np.expand_dims(X[index], 0)
    ground_pain = Y[index]
    #https://stackoverflow.com/questions/1859864/how-to-create-an-integer-array-in-python
    outputs = [0 for i in range(MAX_PAIN)]
    g_vector = gan.Ge.predict(ground_image)

    # ps = pain score
    for ps in range(0, MAX_PAIN):
        ps_arr = np.zeros((1,))
        ps_arr[0] = ps
        newV = np.append(g_vector, np.expand_dims(ps_arr,0), axis=-1)
        outputs[ps] = gan.Gd.predict(newV)
    loss_v = [0 for i in range(MAX_PAIN)]

    #Calculate loss
    for i in range(0, MAX_PAIN):
        #loss_v[i] = sklearn.metrics.mean_squared_error(ground_image, outputs[i])
        #https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy
        loss_v[i] = (np.square(ground_image - outputs[i]).mean(axis=None))

    min_loss = 100
    for i in range(0, MAX_PAIN):
        if loss_v[i] < min_loss:
            min_loss = loss_v[i]
            score = i

    #print(type(ground_pain))
    #print(type(score))
    #print(loss_v)
    #print("confusion[%d][%d]" % (ground_pain,score))
    

    confusion[int(ground_pain)][int(score)] += 1

    if score==ground_pain:
        TP += 1

np.set_printoptions(threshold=sys.maxsize)
for i in range(0, MAX_PAIN):
    for j in range(0, MAX_PAIN):
        confusion[i][j] = confusion[i][j] / float(len(X))

print(confusion)
print("TP: %d" % TP)
accuracy = float(TP/len(X))
print("Accuracy was " + str(accuracy) + "%")

exit()