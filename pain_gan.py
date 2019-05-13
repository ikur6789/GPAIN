'''
Inspiration from: https://github.com/daymos/simple_keras_GAN/blob/master/gan.py
MIT License

Modified Version:
MIT LICENSE
#
# Copyright 2019 Ian Kurzrock
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
'''

""" Simple implementation of Generative Adversarial Neural Network """
import os
import numpy as np


import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Activation, UpSampling2D, Conv2DTranspose, InputLayer, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

import scipy
from PIL import Image
import pandas
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
plt.switch_backend('agg')   # allows code to run without a system DISPLAY

DEBUG_MODE = False
DISPLAY_MODE = False

MAX_PAIN = 16

imagewidth=28 #112
imageheight=28 #112
imagechannels=1

normalizePain = False

#Original last argument was 100
depth = 128
dim = 7
noise_vector_size = 256#dim*dim*depth#100



if DISPLAY_MODE:
    valid_subjects = ['095-tv095', '047-jl047', '109-ib109', '064-ak064']

def load_imgs():
    from os import listdir
    from os.path import isfile, join
    from os.path import isdir

    
    df_List = [] #[['Subject', 'Task', 'Frame', 'Pain', 'Index']]
    baseDir = "/media/Data/UNBC/TrackerOutputPNG"
    painDir = "/media/Data/UNBC/Frame_Labels/PSPI"
    #loadedImages = [baseDir + "/101-mg101/mg101t1aiaff/frame_det_000420.png"]
        
    for f in listdir(baseDir):
        if(isdir(join(baseDir,f))):
            for g in listdir(join(baseDir,f)):
                if(isdir(join(join(baseDir,f),g))):
                    
                    # Load image file paths                    
                    allImageFiles = listdir(join(join(baseDir,f),g))
                    #imageCnt = len(allImageFiles)
                    #imageCntString = str(imageCnt)
                    #padCnt = len(imageCntString)

                    for h in allImageFiles:                        
                        if(isfile(join(join(join(baseDir,f),g),h))): 
                            # Get frame number
                            frameNum = int(h[10:-4])
                            # Get pain filename       
                            frameNumString = ("%03d") % (frameNum + 1)
                            painFilename = g + frameNumString + "_facs.txt"
                            
                            # Open pain score
                            with open(join(join(join(painDir,f),g),painFilename), "r") as t:
                                pain_score = float(t.readline())
                            
                            # Add to total data
                            df_List += [[f, g, frameNum, pain_score, join(join(join(baseDir,f),g),h)]]
                                                        
                            
    df_UNBC = pandas.DataFrame(df_List, columns=['Subject', 'Task', 'Frame', 'Pain', 'Filepath'])

    df_UNBC.set_index(['Filepath'], inplace=True)

    badFiles = []
    
    with open('/media/Data/UNBC/badFiles_UNBC.txt', "r") as f:
        for line in f:
            badFiles += [line.rstrip('\n')]

    badFiles = [baseDir + "/" + f for f in list(badFiles)]

    #for f in list(badFiles):
     #   df_UNBC.drop(f, inplace=True)


    print(df_UNBC)
    return df_UNBC

def get_timestring():
    import datetime
    cTD = datetime.datetime.now()
    timeString = str(cTD.month) + "_" + str(cTD.day) + "_" + str(cTD.year) + "_" + str(cTD.hour) + "_" + str(cTD.minute)

    return timeString

tS = get_timestring()

class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, width=28, height=28, channels=1):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)

        #self.optimizer = Adam()
        #self.optimizer2 = Adam(lr=0.01)

        self.everyoneOptimizer = Adam()
        # 0.0002
        self.learningParams = {"lr" : 0.0002, "beta_1" : 0.5, "beta_2" : 0.999} 
        #learningParams = {"lr" : 0.001, "beta_1" : 0.5, "beta_2" : 0.999} 
        self.encodeLearningParams = self.learningParams 
        
        self.Ge = self.__generator_enc()
        self.Ge.compile(loss='categorical_crossentropy', optimizer=self.__getOptimizer(**self.encodeLearningParams), metrics=['categorical_accuracy'])

        self.Gd = self.__generator_dec()
        self.Gd.compile(loss='categorical_crossentropy', optimizer=self.__getOptimizer(**self.learningParams), metrics=['categorical_accuracy'])

        self.D = self.__discriminator()
        self.D.compile(loss='categorical_crossentropy', optimizer=self.__getOptimizer(**self.learningParams), metrics=['categorical_accuracy'])

        self.Dimg = self.__discriminator_img()
        self.Dimg.compile(loss='categorical_crossentropy', optimizer=self.__getOptimizer(**self.learningParams), metrics=['categorical_accuracy'])

        #self.stacked_generator_discriminator = #self.__stacked_generator_discriminator()
        self.Discriminator_z = self.__discriminator_z()
        self.Discriminator_z.compile(loss='categorical_crossentropy', optimizer=self.__getOptimizer(**self.learningParams), metrics=['categorical_accuracy'])

        # WARNING: Learning rate reset in train_autoencoder!!!!!!!!!!!!!!!!!!
        self.AutoEncoder = self.__AutoEnc_Pain()
        self.AutoEncoder.compile(loss='mean_squared_error', optimizer=self.__getOptimizer(**self.learningParams), metrics=['mean_squared_error'])

        #Disc_Phase is not a new model, so will omit declaring and compiling here
        
        self.Combined_Model = self.__Combined_Model()
        self.Combined_Model.compile(loss='categorical_crossentropy', optimizer=self.__getOptimizer(**self.learningParams), metrics=['categorical_accuracy']) 

        self.Dec_Disc_Only = self.__Dec_Disc_Only()
        self.Dec_Disc_Only.compile(loss='categorical_crossentropy', optimizer=self.__getOptimizer(**self.learningParams), metrics=['categorical_accuracy']) 


        # Theory: this needs to happen AFTER combined model is compiled?
        self.Dimg.trainable = True
        self.Ge.trainable = True
        self.Gd.trainable = True
        
        #self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        #self.Ge.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        #self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def __getOptimizer(self, **kparams):
         return Adam(**kparams)
         # return self.everyoneOptimizer
    #Ge
    def __generator_enc(self):
        """ Declare generator """

        model = Sequential()
        #model.add(Reshape((self.width*self.height*self.channels,), input_shape=(self.height, self.width, self.channels)))
        model.add(InputLayer(input_shape=(self.width, self.height, self.channels)))
        model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=2))
        

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(noise_vector_size, activation=None))

        model.summary()

        return model

    #Gd
    def __generator_dec(self):
        """Decoder Discriminator"""
        dim = 7
        depth = 128
        model = Sequential()
        #Taking vector as input
        model.add(Dense(noise_vector_size+1, input_shape=(noise_vector_size+1,)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

              
        #try dense(256) here and dense(noise_vector)
        model.add(Dense(256))
        #model.add(Dense(noise_vector_size))

        interpol = 'bilinear' #'nearest' #'bilinear'
        

        model.add(Dense(dim*dim*depth))
        model.add(Reshape((dim, dim, depth)))
        model.add(Dropout(0.4))

        model.add(UpSampling2D(interpolation=interpol))
        model.add(Conv2DTranspose(int(depth), 3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(UpSampling2D(interpolation=interpol))
        model.add(Conv2DTranspose(int(depth/2), 3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(int(depth/4), 3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(1,3,padding='same'))
        #model.add(Activation('sigmoid'))
        
        return model
        
    
    #D
    def __discriminator(self):
        """ Declare discriminator """

        model = Sequential()
        model.add(InputLayer(input_shape=(noise_vector_size,)))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(16))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(2, activation='softmax'))
        model.summary()
        

        return model
    

    def __disc_img_first_part(self):
        model = Sequential()
        #model.add(Reshape((self.width*self.height*self.channels,), input_shape=(self.height, self.width, self.channels)))
        model.add(InputLayer(input_shape=(self.width, self.height, self.channels)))
        model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=2))

        model.add(Flatten())

        return model

    #Dimg
    def __discriminator_img(self):
        """ Declare discriminator """

        painV = Input(shape=(1,)) 
        img = Input(shape=(self.height,self.width,self.channels))

        firstModel = self.__disc_img_first_part()

        x = firstModel(img)

        #x = keras.layers.concatenate([x,painV], axis=-1)

        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(2, activation='softmax')(x)

        #model = Model(inputs=[img,painV], outputs=x)
        model = Model(inputs=img, outputs=x)

        return model

    def __discriminator_z(self):

        self.D.trainable = False

        model = Sequential()
        model.add(self.Ge)
        model.add(self.D)

        return model

    def __Dec_Disc_Only(self):
        self.Dimg.trainable = False

        inputVec = Input(shape=(noise_vector_size,))   
        painV = Input(shape=(1,)) 
        painI = Input(shape=(self.height,self.width,1))

        fullVec = keras.layers.concatenate([inputVec,painV], axis=-1)
        gimg = self.Gd(fullVec)
        dinput = gimg # keras.layers.concatenate([gimg,painI], axis=-1)
        out = self.Dimg(dinput)

        model = Model(inputs=[inputVec,painV,painI], outputs=out)

        return model

    #def model2
    def __Disc_Phase(self, g_x_vector, noise_vector, batch):
        #Not to be a seperate model, but will just implement as a function
        self.Ge.trainable = False
        self.D.trainable = False #True Trying to turn off discriminator training

        disc_x = np.concatenate((g_x_vector, noise_vector))
        disc_y = np.concatenate((np.zeros((np.int64(batch/2), 1)), np.ones((np.int64(batch/2), 1))))
        
        loss = self.Gen_Encoder.train_on_batch(disc_x, disc_y)

        return loss


    #def model3
    '''
    def __Genc_Phase(self):
        self.Ge.trainable = True
        self.D.trainable = False


        model = Sequential()      

        model.add(self.Ge)
        model.add(self.D)

        return model
    '''

    def __AutoEnc_Pain(self):

        self.Ge.trainable = True
        self.Gd.trainable = True

        img = Input(shape=(self.height,self.width,self.channels))

        painV = Input(shape=(1,))

        g = self.Ge(img)

        gp = keras.layers.concatenate([g, painV], axis=-1)

        gimg = self.Gd(gp)

        model = Model(inputs=[img, painV], outputs=gimg)

        return model

    def __Combined_Model(self):

        img = Input(shape=(self.height,self.width,self.channels))

        #painI = Input(shape=(self.height,self.width,1))
        
        painV = Input(shape=(1,))

        gimg = self.AutoEncoder([img,painV])

        self.Dimg.trainable = False

        dinput = gimg #keras.layers.concatenate([gimg,painI], axis=-1)

        out = self.Dimg(dinput)

        #model = Model(inputs=[img,painI,painV], outputs=out)
        model = Model(inputs=[img,painV], outputs=out)

        return model

    def train_autoencoder(self, X_train, Y_train, Y_Img_train, epochs=3, batch = 32, save_interval = 25):
        
        # Make data generator
        datagen = ImageDataGenerator(   )                 
        #            rotation_range=5,
        #            width_shift_range=0.05,
        #            height_shift_range=0.05,
        #            horizontal_flip=True)

        # Generator random indices
        random_indices = list(range(0, len(X_train)))
        np.random.seed(42)

        # RESET LEARNING RATE!!!!
        currentLearningRate = K.get_value(self.AutoEncoder.optimizer.lr)
        currentLearningRate = 0.001
        K.set_value(self.AutoEncoder.optimizer.lr, currentLearningRate)
                        
        for cnt in range(epochs+1):

            np.random.shuffle(random_indices)

            batchCnt = int(len(X_train)/np.int64(batch/2))
            batchSize = np.int64(batch/2)
            bIndex = 0
            for legit_images, ground_pain in datagen.flow(X_train, Y_train, batch_size=batchSize):

            # OLD WAYS  
            #for bIndex in range(batchCnt):
                #bStart = batchSize*bIndex
                #bEnd = bStart + batchSize
                #chosenIndices = random_indices[bStart : bEnd]                
                #legit_images = X_train[chosenIndices].reshape(batchSize, self.width, self.height, self.channels)
                #ground_pain = Y_train[chosenIndices]
                # END OLD WAYS
              
                # PHASE 1: Train AUtoEncoder
                aE_Loss = self.AutoEncoder.train_on_batch([legit_images, ground_pain], legit_images)
                
                if bIndex % save_interval == 0:
                    #self.plot_images(save2file=True, epoch=cnt, step=bIndex, legit_images=legit_images, g_x_vector=g_x_vector)
                    self.plot_pain_gen_train(extraname="AE_", save2file=True, epoch=cnt, step=bIndex, legit_images=legit_images)
                          

                print("E: " + str(cnt) + ", B: " + str(bIndex), end=", ")                
                print("AE:", aE_Loss[0], end=", ")                
                print()

                bIndex += 1
                if bIndex >= batchCnt:
                    break

            self.save_model(extraName="AEPHASE_")

            currentLearningRate /= 2.0
            K.set_value(self.AutoEncoder.optimizer.lr, currentLearningRate)
            print("LOSS:", K.get_value(self.AutoEncoder.optimizer.lr))

    def train_decoder_vs_image_discriminator(self, X_train, Y_train, Y_Img_train, epochs=3, batch = 32, save_interval = 25):

        # Generator random indices
        random_indices = list(range(0, len(X_train)))
        np.random.seed(42)
        
                
        for cnt in range(epochs+1):

            np.random.shuffle(random_indices)

            batchCnt = int(len(X_train)/np.int64(batch/2))
            batchSize = np.int64(batch/2)

            for bIndex in range(batchCnt):

                bStart = batchSize*bIndex
                bEnd = bStart + batchSize
                chosenIndices = random_indices[bStart : bEnd]
                
                legit_images = X_train[chosenIndices].reshape(batchSize, self.width, self.height, self.channels)
                ground_pain = Y_train[chosenIndices]
                ground_img_pain = Y_Img_train[chosenIndices]
                
                # PHASE 1: Train discriminator

                noise_vector = np.random.normal(0, 1, (np.int64(batch/2), noise_vector_size))
                random_pain_score = np.random.randint(0, high=MAX_PAIN, size=(batchSize, 1))
                input_vector = np.concatenate([noise_vector, random_pain_score], axis=-1)                                
                recon_images = self.Gd.predict(input_vector)

                pain_score_img = createTiledPain(random_pain_score, self.width, self.height)
                recon_all = recon_images #np.concatenate([recon_images, pain_score_img], axis=-1)

                recon_ground_zeros = np.zeros(shape=(len(recon_all, ))).reshape((-1, 1))
                recon_ground_ones = np.ones(shape=(len(recon_all, ))).reshape((-1, 1))
                recon_ground = np.concatenate([recon_ground_ones, recon_ground_zeros], axis=-1)

                legit_all = legit_images #np.concatenate([legit_images, ground_img_pain], axis=-1)
                legit_ground_zeros = np.zeros((len(legit_all, ))).reshape((-1, 1))
                legit_ground_ones = np.ones((len(legit_all, ))).reshape((-1, 1))
                legit_ground = np.concatenate([legit_ground_zeros, legit_ground_ones], axis=-1)

                disc_input_all = np.concatenate([recon_all, legit_all], axis=0)
                disc_ground_all = np.concatenate([recon_ground, legit_ground], axis=0)

                discImg_Loss = self.Dimg.train_on_batch(disc_input_all, disc_ground_all)

                # PHASE 2: Train decoder
                noise_vector = np.random.normal(0, 1, (batchSize, noise_vector_size))
                random_pain_score = np.random.randint(0, high=MAX_PAIN, size=(batchSize, 1))
                pain_score_img = createTiledPain(random_pain_score, self.width, self.height)
                recon_ground_zeros = np.zeros(shape=(batchSize, )).reshape((-1, 1))
                recon_ground_ones = np.ones(shape=(batchSize, )).reshape((-1, 1))
                deceptive_ground = np.concatenate([recon_ground_zeros, recon_ground_ones], axis=-1)
                
                decode_Loss = self.Dec_Disc_Only.train_on_batch([noise_vector,random_pain_score,pain_score_img], deceptive_ground)

                
                if bIndex % save_interval == 0:
                    #self.plot_images(save2file=True, epoch=cnt, step=bIndex, legit_images=legit_images, g_x_vector=g_x_vector)
                    #self.plot_pain_gen_train(extraname="DecVsDis_", save2file=True, epoch=cnt, step=bIndex, legit_images=legit_images)
                    self.plot_pain_gen_noise(extraname="DecVsDis_", save2file=True, epoch=cnt, step=bIndex)
                
                print("E: " + str(cnt) + ", B: " + str(bIndex), end=", ")                
                print("Dimg:", discImg_Loss, end=", ")
                print("Dec:", decode_Loss, end=", ")
                print()

            self.save_model()

    def train(self, X_train, Y_train, Y_Img_train, epochs=10, batch = 32, save_interval = 25):

        # Generator random indices
        random_indices = list(range(0, len(X_train)))
        np.random.seed(42)

        # Reset AutoEncoder learning rate
        currentLearningRate = self.learningParams["lr"]
        K.set_value(self.AutoEncoder.optimizer.lr, currentLearningRate)
                        
        for cnt in range(epochs+1):

            np.random.shuffle(random_indices)

            batchCnt = int(len(X_train)/np.int64(batch/2))
            batchSize = np.int64(batch/2)

            for bIndex in range(batchCnt):

                bStart = batchSize*bIndex
                bEnd = bStart + batchSize
                chosenIndices = random_indices[bStart : bEnd]
                
                legit_images = X_train[chosenIndices].reshape(batchSize, self.width, self.height, self.channels)
                ground_pain = Y_train[chosenIndices]
                ground_img_pain = Y_Img_train[chosenIndices]

                #Imported from mod_gan3, working on original autoencoder
                ##################################################3
                g_x_vector = self.Ge.predict(legit_images)
                noise_vector = np.random.normal(0, 1, (np.int64(batch/2), noise_vector_size))

                disc_x = np.concatenate((g_x_vector, noise_vector))

                ###
                #Added from face_gan
                halfBatch = np.int64(batch/2)
                enc_ground = np.concatenate([np.ones((halfBatch, 1)), np.zeros((halfBatch, 1))], axis=1)
                noise_ground = np.concatenate([np.zeros((halfBatch, 1)), np.ones((halfBatch, 1))], axis=1)
                disc_y = np.concatenate([enc_ground, noise_ground])
                #
                ###

                #disc_y = np.concatenate((np.zeros((np.int64(batch/2), 1)), np.ones((np.int64(batch/2), 1))))
            
                disc_Loss = self.D.train_on_batch(disc_x, disc_y)
                #gEnc_Loss = self.Gen_Encoder.train_on_batch(legit_images, noise_ground)
                ##################################################
                
                # PHASE 1: Train AUtoEncoder
                aE_Loss = self.AutoEncoder.train_on_batch([legit_images, ground_pain], legit_images)

                # PHASE 2: Train Discrimator

                # Half of the input is legit_images + ground_img_pain
                # Half of the input is gimg + ground_pain      

                recon_all = self.AutoEncoder.predict([legit_images, ground_pain])
                recon_ground_zeros = np.zeros(shape=(len(recon_all, ))).reshape((-1, 1))
                recon_ground_ones = np.ones(shape=(len(recon_all, ))).reshape((-1, 1))
                recon_ground = np.concatenate([recon_ground_ones, recon_ground_zeros], axis=-1)

                legit_all = legit_images # np.concatenate([legit_images, ground_img_pain], axis=-1)
                legit_ground_zeros = np.zeros((len(legit_all, ))).reshape((-1, 1))
                legit_ground_ones = np.ones((len(legit_all, ))).reshape((-1, 1))
                legit_ground = np.concatenate([legit_ground_zeros, legit_ground_ones], axis=-1)

                disc_input_all = np.concatenate([recon_all, legit_all], axis=0)
                disc_ground_all = np.concatenate([recon_ground, legit_ground], axis=0)
          

                discImg_Loss = self.Dimg.train_on_batch(disc_input_all, disc_ground_all)

                # PHASE 3: Train AutoEncoder vs. Discriminator
                
                deceptive_ground = np.concatenate([recon_ground_zeros, recon_ground_ones], axis=-1)
                aEvD_Loss = self.Combined_Model.train_on_batch([legit_images, ground_pain], deceptive_ground)


                #print("NAMES:", self.Dimg.metrics_names, self.Combined_Model.metrics_names)
                #print("GROUND DIMG:", disc_ground_all)
                #print("GROUND COMBO:", deceptive_ground)

                #print("SHAPE GROUND DIMG:", disc_ground_all.shape)
                #print("SHAPE GROUND COMBO:", deceptive_ground.shape)
                
                #dimgPred = self.Dimg.predict(disc_input_all)
                #comboPred = self.Combined_Model.predict([legit_images, ground_pain])
                #print("SHAPE PREDICT DIMG:", dimgPred.shape)
                #print("SHAPE PREDICT COMBO:", comboPred.shape)


                #print("DIMG PREDICT: ", dimgPred)
                #print("COMBO PREDICT: ", comboPred)

                discEval = self.D.evaluate(disc_x, disc_y, verbose=0)
                aeEval = self.AutoEncoder.evaluate([legit_images, ground_pain], legit_images, verbose=0)
                dimgEval = self.Dimg.evaluate(disc_input_all, disc_ground_all, verbose=0)
                comboEval = self.Combined_Model.evaluate([legit_images, ground_pain], deceptive_ground, verbose=0)

                disc_Loss = discEval
                aE_Loss = aeEval
                discImg_Loss = dimgEval
                aEvD_Loss = comboEval                
                
                if bIndex % save_interval == 0:
                    #self.plot_images(save2file=True, epoch=cnt, step=bIndex, legit_images=legit_images, g_x_vector=g_x_vector)
                    self.plot_pain_gen_train(save2file=True, epoch=cnt, step=bIndex, legit_images=legit_images)
                
                print("E: " + str(cnt) + ", B: " + str(bIndex), end=", ")
                print("D: ", disc_Loss, end=", ") 
                print("AE:", aE_Loss[0], end=", ")
                print("Dimg:", discImg_Loss, end=", ")
                print("AEvD:", aEvD_Loss, end=", ")
                print()

            self.save_model()

            currentLearningRate /= 2.0
            K.set_value(self.AutoEncoder.optimizer.lr, currentLearningRate)
            K.set_value(self.D.optimizer.lr, currentLearningRate)
            K.set_value(self.Dimg.optimizer.lr, currentLearningRate)
            K.set_value(self.Combined_Model.optimizer.lr, currentLearningRate)
            print("LOSS:", K.get_value(self.AutoEncoder.optimizer.lr))
    '''
    def test(self, X_test, Y_test, Y_Img_test):

        for index in range(len(X_test)):
            if index % 4 == 0:
                if index == 0:
                    continue
                print("Testing set: " + str(index/4))
                self.plot_pain_gen(save2file=True, epoch=(index/4), legit_images=X_test[index-4:index], ground_pain=Y_test[index-4:index])
                
        return
    '''
    def test(self, X_test, Y_test):

        '''
        legit_images = X_test.reshape(len(X_train), self.width, self.height, self.channels)
        miniCnt = 4
        for index in range(miniCnt, len(legit_images), miniCnt):
            print("Testing set: " + str(index-miniCnt) + " to " + str(index))
            self.plot_pain_gen(save2file=True, epoch=(index-miniCnt), legit_images=legit_images[index-miniCnt:index], ground_pain=Y_test[index-miniCnt:index])
        '''
        miniCnt = 4
        batchCnt = int(len(X_test))
        batchSize = miniCnt

        for index in range(4, batchCnt, 4):

            bStart = batchSize*index
            bEnd = bStart + batchSize
            #chosenIndices = random_indices[bStart : bEnd]
            
            legit_images = X_test[index-miniCnt:index]
            ground_pain = Y_test[index-miniCnt:index]
            #ground_img_pain = Y_Img_test[bStart : bEnd]

            print("Testing set: " + str(index-miniCnt) + " to " + str(index) + "/" + str(len(X_test)))
            self.plot_pain_gen_test(save2file=True, epoch=(index-miniCnt), legit_images=legit_images, ground_pain=ground_pain)

        return
    
    def generate_pain_combination_images(self, legit_images):
        # Run through encoder
        samples = len(legit_images)
        g_vector = self.Ge.predict(legit_images)
        tiled_g_vector = np.tile(g_vector, (MAX_PAIN, 1))

        # Append pain scores and run through decoder        
        pain = np.array(range(0, MAX_PAIN)).reshape((1,MAX_PAIN))        
        if normalizePain:
            pain = np.true_divide(pain, MAX_PAIN)

        # DEBUG: REMOVE THIS LINE!!!!!!
        #pain = np.multiply(pain, 0)


        tiled_pain = np.tile(pain, (samples,1))
        tiled_pain_col = np.transpose(tiled_pain).reshape((samples*MAX_PAIN, 1))
        gp_vector = np.concatenate((tiled_g_vector, tiled_pain_col), axis=1)
        recon_images = self.Gd.predict(gp_vector)

        # Get tiled pain arrays
        pain_score_img = createTiledPain(tiled_pain_col, self.width, self.height)

        return recon_images, pain_score_img, tiled_pain_col
        
    def plot_pain_gen_test(self, save2file=False, epoch=0, samples=4, step=0, legit_images=[], ground_pain=[]):
        ''' Plot and generated images '''
        
        outDir = "./"+tS+"images"
        if not os.path.exists(outDir):
            os.makedirs(outDir)        
        
        filenameRec = "./" + outDir + "/RP_%d.png" % epoch
        
        # Run images through the autoencoder
        recon_images, _, _ = self.generate_pain_combination_images(legit_images[0:samples])        
        recon_images = recon_images.reshape((MAX_PAIN, samples, self.height, self.width, -1))


        imageSize = 4.0
        plt.figure(figsize=(imageSize*MAX_PAIN, (imageSize*samples)))
        plt.margins(0,0)
        #fig, axs = plt.subplots(samples, MAX_PAIN)
        #fig.subplots_adjust(hspace=0, wspace=0)

        gs = gridspec.GridSpec((samples), MAX_PAIN, width_ratios=[1]*MAX_PAIN,
                                    wspace=0.01, hspace=0.01, top=1.0, bottom=0.0, left=0.0, right=1.0) 

        print(samples)
        print(ground_pain.shape)
        imageIndex = 0
        for s in range(samples):
            print(ground_pain[s])
            for p in range(MAX_PAIN):
                #plt.subplot(samples, MAX_PAIN, imageIndex)
                image = recon_images[p, s, :, :, :]
                image = np.reshape(image, [self.height, self.width])

                ax = plt.subplot(gs[s,p])
                ax.imshow(image, cmap='gray', aspect='auto')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                if p == ground_pain[s]:
                    ax.add_patch(patches.Rectangle((0,0), self.width-1, self.height-1, linewidth=5, edgecolor='g', facecolor='none'))
                    ax.axis('on')
                    #ax.spines['bottom'].set_color('green')
                    #ax.spines['left'].set_color('green')
                    #ax.spines['top'].set_color('green')
                    #ax.spines['right'].set_color('green')
                    #ax.spines.facecolor('green')
                else:
                    ax.axis('off')
                
                if s == (samples-1):
                    ax.text(0, self.height-1, str(p), fontsize=48, color='white') 
                
                
                # OLD
                #axs[s, p].imshow(image, cmap='gray')                
                #axs[s, p].axis('off')
                
                imageIndex += 1
        #fig.tight_layout()
        
        '''
        t = samples
        for p in range(MAX_PAIN):
            ax2 = plt.subplot(gs[t,p])
            ax2.text(0, self.height, str(p), fontsize=48, color='white')
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.axis('off')
        '''
        
        if save2file:
            plt.savefig(filenameRec)
            plt.close('all')
        else:
            plt.show()

    def plot_pain_gen_train(self, extraname="", save2file=False, epoch=0, samples=4, step=0, legit_images=[], ground_pain=[]):
        ''' Plot and generated images '''
        
        outDir = "./"+tS+"images"
        if not os.path.exists(outDir):
            os.makedirs(outDir)        
        
        filenameRec = "./" + outDir + "/" + extraname + "RP_%d_%d.png" % (epoch, step)
        
        # Run images through the autoencoder
        recon_images, _, _ = self.generate_pain_combination_images(legit_images[0:samples])        
        recon_images = recon_images.reshape((MAX_PAIN, samples, self.height, self.width, -1))


        imageSize = 4.0
        plt.figure(figsize=(imageSize*MAX_PAIN, imageSize*samples))
        plt.margins(0,0)
        #fig, axs = plt.subplots(samples, MAX_PAIN)
        #fig.subplots_adjust(hspace=0, wspace=0)

        gs = gridspec.GridSpec(samples, MAX_PAIN, width_ratios=[1]*MAX_PAIN,
                                    wspace=0.01, hspace=0.01, top=1.0, bottom=0.0, left=0.0, right=1.0) 

        
        imageIndex = 0
        for s in range(samples):
            for p in range(MAX_PAIN):
                #plt.subplot(samples, MAX_PAIN, imageIndex)
                image = recon_images[p, s, :, :, :]
                image = np.reshape(image, [self.height, self.width])

                ax = plt.subplot(gs[s,p])
                ax.imshow(image, cmap='gray', aspect='auto')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.axis('off')
                
                # OLD
                #axs[s, p].imshow(image, cmap='gray')                
                #axs[s, p].axis('off')
                
                imageIndex += 1
        #fig.tight_layout()

        if save2file:
            plt.savefig(filenameRec)
            plt.close('all')
        else:
            plt.show()

    def plot_pain_gen_noise(self, extraname="", save2file=False, epoch=0, samples=4, step=0):
        ''' Plot and generated images '''
        
        outDir = "./"+tS+"images"
        if not os.path.exists(outDir):
            os.makedirs(outDir)        
        
        filenameRec = "./" + outDir + "/" + extraname + "RP_%d_%d.png" % (epoch, step)
        
        # Run noise through the decoder
        noise_vector = np.random.normal(0, 1, (samples, noise_vector_size))
        tiled_g_vector = np.tile(noise_vector, (MAX_PAIN, 1))
        pain = np.array(range(0, MAX_PAIN)).reshape((1,MAX_PAIN))        
        if normalizePain:
            pain = np.true_divide(pain, MAX_PAIN)
        tiled_pain = np.tile(pain, (samples,1))
        tiled_pain_col = np.transpose(tiled_pain).reshape((samples*MAX_PAIN, 1))
        gp_vector = np.concatenate((tiled_g_vector, tiled_pain_col), axis=1)
        recon_images = self.Gd.predict(gp_vector)

        recon_images = recon_images.reshape((MAX_PAIN, samples, self.height, self.width, -1))

        imageSize = 4.0
        plt.figure(figsize=(imageSize*MAX_PAIN, imageSize*samples))
        plt.margins(0,0)
        #fig, axs = plt.subplots(samples, MAX_PAIN)
        #fig.subplots_adjust(hspace=0, wspace=0)

        gs = gridspec.GridSpec(samples, MAX_PAIN, width_ratios=[1]*MAX_PAIN,
                                    wspace=0.01, hspace=0.01, top=1.0, bottom=0.0, left=0.0, right=1.0) 

        
        imageIndex = 0
        for s in range(samples):
            for p in range(MAX_PAIN):
                #plt.subplot(samples, MAX_PAIN, imageIndex)
                image = recon_images[p, s, :, :, :]
                image = np.reshape(image, [self.height, self.width])

                ax = plt.subplot(gs[s,p])
                ax.imshow(image, cmap='gray', aspect='auto')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.axis('off')
                
                # OLD
                #axs[s, p].imshow(image, cmap='gray')                
                #axs[s, p].axis('off')
                
                imageIndex += 1
        #fig.tight_layout()

        if save2file:
            plt.savefig(filenameRec)
            plt.close('all')
        else:
            plt.show()


    def plot_images(self, save2file=False, epoch=0, samples=16, step=0, legit_images=[], g_x_vector=[]):
        ''' Plot and generated images '''
        
        outDir = "./"+tS+"images"
        if not os.path.exists(outDir):
            os.makedirs(outDir)        
        filenameNoise = "./" + outDir + "/N_%d_%d.png" % epoch, step
        filenameRec = "./" + outDir + "/R_%d_%d.png" % epoch, step
        #noise = np.random.normal(0, 1, (samples, noise_vector_size))

        # Images from noise
        noise = np.random.normal(0, 1, (samples, noise_vector_size))
        noiseImages = self.Gd.predict(noise)

        g_vector = self.Ge.predict(legit_images[0:samples])
        reconImages = self.Gd.predict(g_vector)

        plt.figure(figsize=(10, 10))

        for i in range(noiseImages.shape[0]):
            plt.subplot(4, 4, i+1)
            image = noiseImages[i, :, :, :]
            image = np.reshape(image, [self.height, self.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()

        if save2file:
            plt.savefig(filenameNoise)
            plt.close('all')
        else:
            plt.show()

        for i in range(reconImages.shape[0]):
            plt.subplot(4, 4, i+1)
            image = reconImages[i, :, :, :]
            image = np.reshape(image, [self.height, self.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()

        if save2file:
            plt.savefig(filenameRec)
            plt.close('all')
        else:
            plt.show()

    def save_model(self, extraName=""):

        outDir = "./"+tS+"model"
        if not os.path.exists(outDir):
            os.makedirs(outDir)   

        self.Ge.save_weights(outDir + "/" + extraName + "model_Ge.h5")
        self.Gd.save_weights(outDir + "/" + extraName + "model_Gd.h5")
        self.D.save_weights(outDir + "/" + extraName + "model_D.h5")
        self.Dimg.save_weights(outDir + "/" + extraName + "model_Dimg.h5") 
        self.Discriminator_z.save_weights(outDir + "/" + extraName + "model_Discriminator_z.h5")        
        self.AutoEncoder.save_weights(outDir + "/" + extraName + "model_AutoEncoder.h5")        
        self.Combined_Model.save(outDir + "/" + extraName + "model_Combined_Model.h5")

        print("Saved weights.")

    def load_model(self, foldername, extraName=""):
       
        self.Ge.load_weights(foldername + "/" + extraName + "model_Ge.h5")
        self.Gd.load_weights(foldername + "/" + extraName + "model_Gd.h5")
        self.D.load_weights(foldername + "/" + extraName + "model_D.h5")
        self.Dimg.load_weights(foldername + "/" + extraName + "model_Dimg.h5") 
        self.Discriminator_z.load_weights(foldername + "/" + extraName + "model_Discriminator_z.h5")        
        self.AutoEncoder.load_weights(foldername + "/" + extraName + "model_AutoEncoder.h5")        
        #self.Combined_Model = load_model(foldername + "/" + extraName + "model_Combined_Model.h5")

        print("Loaded weights.")
        

def createTiledPain(Y, width, height):
    Y = np.copy(Y)
    #numpy.set_printoptions(threshold=sys.maxsize)
    Y = np.reshape(Y, (-1, 1, 1, 1))
    Y = np.tile(Y, (1, height, width, 1))
    return Y

if __name__ == '__main__':
    #(X_train, _), (_, _) = mnist.load_data()

    if DEBUG_MODE:
        print("************************************", file=sys.stderr)
        print("************************************", file=sys.stderr)
        print("WARNING: IN DEBUG MODE!!!!!", file=sys.stderr)
        print("************************************", file=sys.stderr)
        print("************************************", file=sys.stderr)

    
    
    all_Data = load_imgs()




    #print(all_Data['Filepath'].values)

    

    allSubjects = list(set(all_Data['Subject'].values))
    random_subject_indices = list(range(0, len(allSubjects)))
    np.random.seed(42)
    np.random.shuffle(random_subject_indices)
    split_percent = 0.70
    split_cnt = int(split_percent*len(allSubjects))
    train_subject_indices = random_subject_indices[0:split_cnt]
    test_subject_indices = random_subject_indices[split_cnt+1:]
    train_subjects = np.array(allSubjects)[train_subject_indices]
    test_subjects = np.array(allSubjects)[test_subject_indices]

    # DEBUG
    if DEBUG_MODE:
        train_subjects = train_subjects[0:2]
    # END DEBUG

    

    train_Data = all_Data[all_Data['Subject'].isin(train_subjects)]
    test_Data = all_Data[all_Data['Subject'].isin(test_subjects)]

    print("Training subjects:", list(train_subjects))
    print("Number", str(len(train_subjects)))
    print("Testing subjects:", list(test_subjects))
    print("Number", str(len(test_subjects)))

    print(all_Data)
    X_train = [scipy.ndimage.imread(f, flatten=True) for f in list(train_Data.index)]
    Y_train = train_Data['Pain'].values
    if normalizePain:
        Y_train = np.true_divide(Y_train, MAX_PAIN)

    Y_Img_train = createTiledPain(Y_train, imagewidth, imageheight)

    X_test = [scipy.ndimage.imread(f, flatten=True) for f in list(test_Data.index)]
    Y_test = train_Data['Pain'].values
    if normalizePain:
        Y_test = np.true_divide(Y_train, MAX_PAIN)

    Y_Img_test = createTiledPain(Y_test, imagewidth, imageheight)

   
    # DEBUG:
    #X_train = [scipy.ndimage.imread(f, flatten=True) for f in list(all_Data['Filepath'].values)]
    #Y_train = all_Data['Pain'].values
    


    #res = df[df['Channel'].isin({'A', 'B'})]
   
    #X_train =[X_train]
    
    for i in range (len(X_train)) :
        tempImage=Image.fromarray(X_train[i])
        X_train[i]=np.array(tempImage.resize(size=(imageheight,imagewidth)))

    for i in range (len(X_test)) :
        tempImage=Image.fromarray(X_test[i])
        X_test[i]=np.array(tempImage.resize(size=(imageheight,imagewidth)))

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=3)

    gan = GAN(width=imagewidth, height=imageheight, channels=imagechannels)

    #gan.test(X_test, Y_test, Y_Img_test)

    #gan.train(X_train, Y_train, Y_Img_train)
    
    #gan.train_autoencoder(X_train, Y_train, Y_Img_train, epochs=10)
    gan.train(X_train, Y_train, Y_Img_train)
    
    #gan.train_decoder_vs_image_discriminator(X_train, Y_train, Y_Img_train)


    print("****************TEST PHASE*******************")
    gan.test(X_test, Y_test)