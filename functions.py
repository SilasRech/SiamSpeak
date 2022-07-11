import numpy as np
import argparse
from numpy import *
from sklearn.utils import shuffle
import pickle
import warnings
from random import randint

from sklearn.metrics.pairwise import cosine_similarity

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, Flatten, MaxPooling2D, Reshape, concatenate, BatchNormalization, Dropout, Activation
from keras import optimizers
from keras import initializers

def cnn(opt, n_blocks, n_filters, input_shape):
    print('.... Constructing CNN')
    init=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    inputs = Input(shape=(input_shape))

    if len(n_filters) < n_blocks:
       print('..... Define filter size for every CNN block')
    else:
       for i in range(n_blocks):
         name = ('block%d_conv_1'%(i+1))
         if i == 0:
            x = Conv2D(n_filters[i], kernel_size=(3,3), padding="same", activation=opt.activation_function, name=name)(inputs)
         else:
            x = Conv2D(n_filters[i], kernel_size=(3,3), padding="same", activation=opt.activation_function, name=name)(x)
         name = ('block%d_conv_2'%(i+1))
         x = Conv2D(n_filters[i], kernel_size=(3,3), padding="same", activation=opt.activation_function, name=name)(x)
         name = ('block%d_pool'%(i+1))
         x = MaxPooling2D(pool_size=(2,2), name=name)(x)
    x = Flatten()(x)
    x = Dense(1000, activation=opt.activation_function)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(400, activation=opt.activation_function, name='Embedding')(x)
    x = Dense(opt.n_classes, activation='softmax', name='Classification')(x)
    model = Model(inputs, x)
    return model

def normalize(features):
    np.seterr(divide='ignore', invalid='ignore')
    mean = np.mean(features,axis=0)
    std = np.std(features, axis=0)
    features = (features-mean)/std
    return features

def get_vector(Features, window_size):
    Features = normalize(Features)
    file_size = Features.shape[0]
    index = randint(0, max(0,file_size-window_size-1))
    a = np.array(range(min(file_size, window_size)))+index
    new_vec = Features[a,:]
    return np.expand_dims(new_vec, axis=-1)

def read_file(list):
    name = []
    label = []
    lines = list.readlines()
    for x in lines:
        name.append(x.split()[0])
        label.append(x.split()[1])
    return name, label

def read_trials(list):
    name1 = []
    name2 = []
    label = []
    lines = list.readlines()
    for x in lines:
        name1.append(x.split()[0])
        name2.append(x.split()[1])
        label.append(x.split()[2])
    return name1, name2, label

def load_data(opt, data_names):
    print('.... Loading data')
    data = []
    for i in range(len(data_names)):
        with open(opt.spec_path + data_names[i] + '.pickle', 'rb') as f:
             data.append(get_vector(np.transpose(pickle.load(f)), opt.window_size))
    return data

def predict_by_model(opt, dnn, val_names, score_file, layer_name):
    model = Model(inputs=dnn.input, outputs=dnn.get_layer(layer_name).output)
    #print(model.summary())
    scores = []
    for i in range(len(val_names[0])):
        with open(opt.spec_path + val_names[0][i] + '.pickle', 'rb') as f:
             sample1 = get_vector(np.transpose(pickle.load(f)), opt.window_size)
             sample1 = np.expand_dims(sample1,axis=0)
        with open(opt.spec_path + val_names[1][i] + '.pickle', 'rb') as f:
             sample2 = get_vector(np.transpose(pickle.load(f)), opt.window_size)
             sample2 = np.expand_dims(sample2,axis=0)
        sample1 = model.predict([sample1])
        sample2 = model.predict([sample2])
        scores.append(np.squeeze(cosine_similarity(sample1, sample2)))
        #print('%d %s' %(i,scores[i]))
    scores = np.array(scores)
    out = open(score_file, 'w')
    for a in scores:
        out.write('%s\n'%a)
    out.close()
    print('..... Scores are written in: %s' %score_file)
