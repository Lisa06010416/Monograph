#!/usr/bin/env python
# coding: utf-8

# In[41]:


import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Lambda
import keras.backend as K
from keras.layers import Concatenate
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dense
from keras.layers import Input
import numpy as np
import os
import random
from PIL import Image
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Dropout
import multiprocessing as mp

input_shape=(224, 224, 3)
embeddingSize=128
# Data preprocess
# contain split train vali
# read img
# generalTrainSample
class Data():
    def __init__(self, path):
        self.path = path
        self.Person, self.picture_perPerson = self.getAllImgDic()
    
    def splitData(self, ratio):
        # split train and validation
        splitIndex = int(np.ceil(len(self.Person)*ratio))
        self.trainIndex = range(0,splitIndex)
        self.validationIndex = range(splitIndex,len(self.Person))

    def generateValSet(self, num):
        # random select validation set
        validationSet = []
        for i in range(num):
            triplet = []
            # random sample 2 person
            person1, person2 = random.sample(self.validationIndex, 2)
            # random sample 2 picture(anchor, pos) from person1            
            pa, pp = random.sample(self.picture_perPerson[person1], 2)
            anchotPath = self.path+"/"+self.Person[person1]+"/"+pa
            posPath = self.path+"/"+self.Person[person1]+"/"+pp
            # random sample 1 picture(neg) from person2
            pn = random.sample(self.picture_perPerson[person2], 1)[0]
            negPath = self.path+"/"+self.Person[person2]+"/"+pn
            # append triplet into validationSet
            validationSet.append([anchotPath, posPath, negPath])
            
        self.validationSet = validationSet
        
    def getAllImgDic(self):        
        ALL_picturePath_perPerson = []
        person_folderList = []
        for person in os.listdir(self.path):  # get all picture under every person
            person_path = self.path + "/" + person
            picture_List = os.listdir(person_path)
            if(len(picture_List)!=0):
                person_folderList.append(person)
                ALL_picturePath_perPerson.append(picture_List)
        return person_folderList, ALL_picturePath_perPerson
    
    def startPoolReadImgList(self, imgPathList):
        print("start pool")     
        P = mp.Pool(processes=16)
        imgList = P.map(self.readImgList, imgPathList)
        return np.array(imgList)

        # get all path of image under dataset
    def readImgList(self,imgPathList):  
        img = Image.open(imgPathList)
        img = img.resize((224, 224))
        return np.array(img)
    
    
                    
    def generalTrainSample(self,num):
        # 1. from Person list random select pictures(ancher and pos) which number is batchsize
        trainSet = []
        for i in range(num):
            triplet = []
            # random sample 2 person
            person1, person2 = random.sample(self.trainIndex, 2)
            # random sample 2 picture(anchor, pos) from person1            
            pa, pp = random.sample(self.picture_perPerson[person1], 2)
            anchotPath = self.path+"/"+self.Person[person1]+"/"+pa
            posPath = self.path+"/"+self.Person[person1]+"/"+pp
            # random sample 1 picture(neg) from person2
            pn = random.sample(self.picture_perPerson[person2], 1)[0]
            negPath = self.path+"/"+self.Person[person2]+"/"+pn
            # append triplet into validationSet
            trainSet.append([anchotPath, posPath, negPath])
        return trainSet


# ### facenet model

# In[6]:
def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1      
        x = BatchNormalization(axis=bn_axis, momentum=0.995, epsilon=0.001,
                               scale=False, name=name+"_bn")(x)
    x = Activation(activation)(x)
    return x


def l2_norm(x):
    return  K.l2_normalize(x, axis=1)

def reshape(x):
    return  tf.reshape(x,(embeddingSize,))
# In[ ]:


def euclidean_distance(vects):
    x, y = vects
#     return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    return  K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


# In[ ]:


def triplet_loss(y_true, y_pred):
    margin = K.constant(4)
#     return  K.mean(y_pred[:,0,0] - y_pred[:,1,0])
    return K.mean(K.maximum(K.constant(0), y_pred[:,0,0] - y_pred[:,1,0] + margin))
#     return K.mean(K.maximum(K.constant(0), K.square(y_pred[0]) - (K.square(y_pred[1])-K.square(y_pred[2])) + margin))


# In[ ]:


def myAccuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])


# In[ ]:


def facenetModel(batchSize,
                    dropout_keep_prob=0.8,
                  weights_path=None,
                ):
    
    inputs = Input(shape=input_shape)
    channel_axis = 3
    # conv1 (7×7×3, 2)
    x = conv2d_bn(inputs, 64, 7, strides=(2, 2), padding='same', name='Conv1_7')
    x = MaxPooling2D(3, strides=(2, 2), padding='same', name='MaxPool_3_1')(x)
#     x = Lambda(l2_norm)(x)
    
    # inception (2)
    x = conv2d_bn(x, 64, 1, strides=(1,1), padding='same', name='inception2_1')
    x = conv2d_bn(x, 192, 3, strides=(1,1), padding='same', name='inception2_3')
    
    # norm + max pool
#     x = Lambda(l2_norm)(x)
    x = MaxPooling2D(3, strides=(2,2), padding='same', name='MaxPool_3x3_2')(x)
    
    # inception (3a)
    branch_0 = conv2d_bn(x,64, 1, strides=(1,1), padding='same', name='inception_3a_1')
    branch_1 = conv2d_bn(x,96, 1, strides=(1,1), padding='same', name='inception_3a_3_re')
    branch_1 = conv2d_bn(branch_1,128, 3, strides=(1,1), padding='same', name='inception_3a_3')
    branch_2 = conv2d_bn(x,16, 1, strides=(1,1), padding='same', name='inception_3a_5_re')
    branch_2 = conv2d_bn(branch_2,32, 5, strides=(1,1), padding='same', name='inception_3a_5')
    branch_3 = MaxPooling2D(3, strides=(1,1), padding='same', name='inception_3a_mp')(x)
    branch_3 = conv2d_bn(branch_3,32, 1, strides=(1,1), padding='same', name='inception_3a_mp_1')
    branches = [branch_0, branch_1, branch_2, branch_3]
    x = Concatenate(axis=channel_axis, name='Mixed_3a')(branches)
    
    # inception (3b)
    branch_0 = conv2d_bn(x,64, 1, strides=(1,1), padding='same', name='inception_3b_1')
    branch_1 = conv2d_bn(x,96, 1, strides=(1,1), padding='same', name='inception_3b_3_re')
    branch_1 = conv2d_bn(branch_1,128, 3, strides=(1,1), padding='same', name='inception_3b_3')
    branch_2 = conv2d_bn(x,32, 1, strides=(1,1), padding='same', name='inception_3b_5_re')
    branch_2 = conv2d_bn(branch_2,64, 5, strides=(1,1), padding='same', name='inception_3b_5')
    branch_3 = MaxPooling2D(3, strides=(1, 1), padding='same', name='inception_3b_mp')(x)
    branch_3 = conv2d_bn(branch_3,64, 1, strides=(1,1), padding='same', name='inception_3b_n_1')
    branches = [branch_0, branch_1, branch_2, branch_3]
    x = Concatenate(axis=channel_axis, name='Mixed_3b')(branches)
    
    # inception (3c)
    branch_1 = conv2d_bn(x,128, 1, strides=(1,1), padding='same', name='inception_3c_3_re')
    branch_1 = conv2d_bn(branch_1,256, 3, strides=(2, 2), padding='same', name='inception_3c_3')
    branch_2 = conv2d_bn(x,32, 1, strides=(1,1), padding='same', name='inception_3c_5_re')
    branch_2 = conv2d_bn(branch_2,64, 5, strides=(2, 2), padding='same', name='inception_3c_5')
    branch_3 = MaxPooling2D(3, strides=(2, 2), padding='same', name='inception_3c_mp')(x)
    branches = [branch_1, branch_2, branch_3]
    x = Concatenate(axis=channel_axis, name='Mixed_3c')(branches)
    
    # inception (4a)
    branch_0 = conv2d_bn(x,256, 1, strides=(1,1), padding='same', name='inception_4a_1')
    branch_1 = conv2d_bn(x,96, 1, strides=(1,1), padding='same', name='inception_4a_3_re')
    branch_1 = conv2d_bn(branch_1,192, 3, strides=(1,1), padding='same', name='inception_4a_3')
    branch_2 = conv2d_bn(x,32, 1, strides=(1,1), padding='same', name='inception_4a_5_re')
    branch_2 = conv2d_bn(branch_2,64, 5, strides=(1,1), padding='same', name='inception_4a_5')
    branch_3 = MaxPooling2D(3, strides=(1, 1), padding='same', name='inception_4a_mp')(x)
    branch_3 = conv2d_bn(branch_3,128, 1, strides=(1,1), padding='same', name='inception_4a_n_1')
    branches = [branch_0, branch_1, branch_2, branch_3]
    x = Concatenate(axis=channel_axis, name='Mixed_4a')(branches)
    
    # inception (4b) 
    branch_0 = conv2d_bn(x,224, 1, strides=(1,1), padding='same', name='inception_4b_1')
    branch_1 = conv2d_bn(x,112, 1, strides=(1,1), padding='same', name='inception_4b_3_re')
    branch_1 = conv2d_bn(branch_1,224, 3, strides=(1,1), padding='same', name='inception_4b_3')
    branch_2 = conv2d_bn(x,32, 1, strides=(1,1), padding='same', name='inception_4b_5_re')
    branch_2 = conv2d_bn(branch_2,64, 5, strides=(1,1), padding='same', name='inception_4b_5')
    branch_3 = MaxPooling2D(3, strides=(1, 1), padding='same', name='inception_4b_mp')(x)
    branch_3 = conv2d_bn(branch_3, 128, 1, strides=(1,1), padding='same', name='inception_4b_n_1')
    branches = [branch_0, branch_1, branch_2, branch_3]
    x = Concatenate(axis=channel_axis, name='Mixed_4b')(branches)
    
    # inception (4c) 
    branch_0 = conv2d_bn(x,192, 1, strides=(1,1), padding='same', name='inception_4c_1')
    branch_1 = conv2d_bn(x,128, 1, strides=(1,1), padding='same', name='inception_4c_3_re')
    branch_1 = conv2d_bn(branch_1, 256, 3, strides=(1,1), padding='same', name='inception_4c_3')
    branch_2 = conv2d_bn(x,32, 1, strides=(1,1), padding='same', name='inception_4c_5_re')
    branch_2 = conv2d_bn(branch_2, 64, 5, strides=(1,1), padding='same', name='inception_4c_5')
    branch_3 = MaxPooling2D(3, strides=(1,1), padding='same', name='inception_4c_mp')(x)
    branch_3 = conv2d_bn(branch_3,128, 1, strides=(1,1), padding='same', name='inception_4c_n_1')
    branches = [branch_0, branch_1, branch_2, branch_3]
    x = Concatenate(axis=channel_axis, name='Mixed_4c')(branches)
    
    # inception (4d) 
    branch_0 = conv2d_bn(x,160, 1, strides=(1,1), padding='same', name='inception_4d_1')
    branch_1 = conv2d_bn(x,114, 1, strides=(1,1), padding='same', name='inception_4d_3_re')
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=(1,1), padding='same', name='inception_4d_3')
    branch_2 = conv2d_bn(x,32, 1, strides=(1,1), padding='same', name='inception_4d_5_re')
    branch_2 = conv2d_bn(branch_2, 64, 5, strides=(1,1), padding='same', name='inception_4d_5')
    branch_3 = MaxPooling2D(3, strides=(1, 1), padding='same', name='inception_4d_mp')(x)
    branch_3 = conv2d_bn(branch_3,128, 1, strides=(1,1), padding='same', name='inception_4d_n_1')
    branches = [branch_0, branch_1, branch_2, branch_3]
    x = Concatenate(axis=channel_axis, name='Mixed_4d')(branches)
    
    # inception (4e)
    branch_1 = conv2d_bn(x,160, 1, strides=(1,1), padding='same', name='inception_4e_3_re')
    branch_1 = conv2d_bn(branch_1, 256, 3, strides=(2, 2), padding='same', name='inception_4e_3')
    branch_2 = conv2d_bn(x,64, 1, strides=(1,1), padding='same', name='inception_4e_5_re')
    branch_2 = conv2d_bn(branch_2,128, 5, strides=(2, 2), padding='same', name='inception_4e_5')
    branch_3 = MaxPooling2D(3, strides=(2, 2), padding='same', name='inception_4e_mp')(x)
    branches = [branch_1, branch_2, branch_3]
    x = Concatenate(axis=channel_axis, name='Mixed_4e')(branches)
    
    # inception (5a)
    branch_0 = conv2d_bn(x,384, 1, strides=(1,1), padding='same', name='inception_5a_1')
    branch_1 = conv2d_bn(x,192, 1, strides=(1,1), padding='same', name='inception_5a_3_re')
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=(1,1), padding='same', name='inception_5a_3')
    branch_2 = conv2d_bn(x,48, 1, strides=(1,1), padding='same', name='inception_5a_5_re')
    branch_2 = conv2d_bn(branch_2,128, 5, strides=(1,1), padding='same', name='inception_5a_5')
    branch_3 = MaxPooling2D(3, strides=(1, 1), padding='same', name='inception_5a_mp')(x)
    branch_3 = conv2d_bn(branch_3, 128, 1, strides=(1,1), padding='same', name='inception_5a_n_1')
    branches = [branch_0, branch_1, branch_2, branch_3]
    x = Concatenate(axis=channel_axis, name='Mixed_5a')(branches)
    
    # inception (5b)
    branch_0 = conv2d_bn(x,384, 1, strides=(1,1), padding='same', name='inception_5b_1')
    branch_1 = conv2d_bn(x,192, 1, strides=(1,1), padding='same', name='inception_5b_3_re')
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=(1,1), padding='same', name='inception_5b_3')
    branch_2 = conv2d_bn(x,48, 1, strides=(1,1), padding='same', name='inception_5b_5_re')
    branch_2 = conv2d_bn(branch_2,128, 5, strides=(1,1), padding='same', name='inception_5b_5')
    branch_3 = MaxPooling2D(3, strides=(1,1), padding='same', name='inception_5b_mp')(x)
    branch_3 = conv2d_bn(branch_3, 128, 1, strides=(1,1), padding='same', name='inception_5b_n_1')
    branches = [branch_0, branch_1, branch_2, branch_3]
    x = Concatenate(axis=channel_axis, name='Mixed_5b')(branches)
    
    # avg pool
    x = AveragePooling2D(7, strides=(1,1), padding='valid', name='AvgPool')(x)
    x = Reshape((1024,), name="Reshape")(x)
    
    # fully conn
    # x = Dropout(0.4)(x)
    x = Dense(embeddingSize, name='FullyConn3')(x)
    
    # bug1 : Output tensors to a Model must be the output of a Keras `Layer`
    # so I can not use K.l2_normalize because it return a tensor not a layer
    # I use a Lambda to do l2_normalize
    # bug2 : name can't be 1x1
    model = Model(inputs, x, name='facenet')
    
    return model


# In[ ]:


def createModel(batchSize,lr):
    facenet = facenetModel(batchSize) #bulid facanet model
    
#     input_shape=(224, 224, 3)
    
    input_anchor = Input(shape=input_shape, name='input_anchor')
    input_positive = Input(shape=input_shape, name='input_pos')
    input_negative = Input(shape=input_shape, name='input_neg')
    
    # reuse model caiculate anchor positive negative sample
    net_anchor = facenet(input_anchor)
    net_positive = facenet(input_positive)
    net_negative = facenet(input_negative)
    
    # caiculate distance
    positive_dist = Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
    negative_dist = Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])
#     tertiary_dist = Lambda(euclidean_distance, name='ter_dist')([net_positive, net_negative])
    
    # This lambda layer simply stacks outputs so both distances are available to the objective
#     stacked_dists = Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist, tertiary_dist])
    stacked_dists = Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist])
    model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')
    
    from keras import optimizers
    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam', loss=triplet_loss, metrics=[myAccuracy])
    
    return facenet, model

