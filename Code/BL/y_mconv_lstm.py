#****************************
# tanfei2007@gmail.com
#****************************


#****import modules****
import h5py
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import roc_auc_score, matthews_corrcoef, \
precision_recall_fscore_support, average_precision_score, accuracy_score
from sklearn import cross_validation
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss


import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from keras.layers import Input, concatenate, TimeDistributed, Reshape,multiply, Masking
from keras.models import Sequential, load_model, Model
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.preprocessing import sequence
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import regularizers
import keras.backend as K

from utility import *
#**********************

k = float(sys.argv[1])
print('k:%s' % k)

feature = ['msno', 'X', 'y', 'aux_s', 'aux_d', 'mask', 'm_y']
data = read_hdf5('0_train.hdf5', feature)
id = data[0]
X = data[1]
y = data[2]
aux_s = data[3]
aux_d = data[4]
mask = data[5]
mask = mask.reshape((mask.shape[0], mask.shape[1]))
m_y = data[6]
print 'loaded done'
y = np.concatenate((m_y, y), axis = 1)
m_y = np.concatenate((np.zeros((m_y.shape[0], 1, 1)), m_y), axis = 1)

"""idx_pos = np.where(y == 1)[0]
idx_neg = np.where(y == 0)[0]
idx_neg = idx_neg[0:idx_pos.shape[0]]
idx = np.concatenate((idx_pos, idx_neg))

y = y[idx]
X = X[idx,:]
id = id[idx]"""


#split data into train, valid, test
train_ratio = 0.6
valid_ratio = 0.9

#X
X_train, X_valid_test, y_train, y_valid_test = cross_validation.train_test_split(X, y,
                                               test_size=1 - train_ratio, random_state=0, stratify=y[:,-1,0])
X_valid, X_test, y_valid, y_test = cross_validation.train_test_split(X_valid_test, y_valid_test,
                                               test_size=1 - valid_ratio, random_state=0, stratify=y_valid_test[:,-1,0])

#aux_s
aux_s_train, aux_s_valid_test, y_train, y_valid_test = cross_validation.train_test_split(aux_s, y,
                                               test_size=1 - train_ratio, random_state=0, stratify=y[:, -1, 0])
aux_s_valid, aux_s_test, y_valid, y_test = cross_validation.train_test_split(aux_s_valid_test, y_valid_test,
                                               test_size=1 - valid_ratio, random_state=0, stratify=y_valid_test[:, -1, 0])

#aux_d
aux_d_train, aux_d_valid_test, y_train, y_valid_test = cross_validation.train_test_split(aux_d, y,
                                               test_size=1 - train_ratio, random_state=0, stratify=y[:, -1, 0])
aux_d_valid, aux_d_test, y_valid, y_test = cross_validation.train_test_split(aux_d_valid_test, y_valid_test,
                                        test_size=1 - valid_ratio, random_state=0, stratify=y_valid_test[:, -1, 0])

#mask
mask_train, mask_valid_test, y_train, y_valid_test = cross_validation.train_test_split(mask, y,
                                               test_size=1 - train_ratio, random_state=0, stratify=y[:, -1, 0])
mask_valid, mask_test, y_valid, y_test = cross_validation.train_test_split(mask_valid_test, y_valid_test,
                                               test_size=1 - valid_ratio, random_state=0, stratify=y_valid_test[:, -1, 0])

#m_y
m_y_train, m_y_valid_test, y_train, y_valid_test = cross_validation.train_test_split(m_y, y,
                                               test_size=1 - train_ratio, random_state=0, stratify=y[:, -1, 0])
m_y_valid, m_y_test, y_valid, y_test = cross_validation.train_test_split(m_y_valid_test, y_valid_test,
                                               test_size=1 - valid_ratio, random_state=0, stratify=y_valid_test[:, -1, 0])

print_id_label('train', id, y_train)
print_id_label('valid', id, y_valid)
print_id_label('test', id, y_test)
#**********************

#****model building****

#input layers
main_input = Input(shape=(X_train.shape[1], X_train.shape[2]), name='main_input')
s_input = Input(shape=(aux_s_train.shape[1], ), name='s_input')
d_input = Input(shape=(aux_d_train.shape[1], aux_d_train.shape[2]), name='d_input')
status_input = Input(shape=(m_y_train.shape[1], m_y_train.shape[2]), name='status_input')

#main path
main_path = Conv1D(filters=7, kernel_size=30, strides = 30, padding='valid', activation = 'relu')(main_input)
#main_path = Conv1D(filters=7, kernel_size=2, strides = 2, padding='valid', activation = 'relu')(main_input)
main_path = concatenate([main_path, status_input], axis = -1)
main_path = LSTM(units=30, return_sequences = True)(main_path)
main_path = LSTM(units=15, return_sequences = True)(main_path)


#s path
s_path = Dense(units=30, activation='relu')(s_input)
s_path = Dense(units=15, activation='relu')(s_path)


#d path
d_path = d_input
d_path = concatenate([d_path, status_input], axis = -1)
d_path =TimeDistributed(Dense(units=30, activation='relu'))(d_path)
d_path =TimeDistributed(Dense(units=15, activation='relu'))(d_path)


#merge path
s_path = Reshape((1, K.int_shape(s_path)[1]))(s_path)
s_paths = concatenate([s_path]*24, axis = 1)
merge_path = concatenate([main_path, s_paths, d_path], axis = -1)

merge_path = TimeDistributed(Dense(units=30, kernel_initializer='he_normal', activation = 'relu'))(merge_path)
merge_path = TimeDistributed(Dense(units=15, kernel_initializer='he_normal', activation = 'relu'))(merge_path)

output = TimeDistributed(Dense(units=1, activation='sigmoid', name='main_output'))(merge_path)
model = Model(inputs = [main_input, s_input, d_input, status_input], outputs = output)
#*******************************


#****model compiling****
adam = Adam(lr=0.001, decay=1e-03)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'], sample_weight_mode = 'temporal')
checkpointer = ModelCheckpoint(filepath='y_mconv_lstm.hdf5' + str(k), verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
#***********************


#****model fit****
print(model.summary())


#set up decay factor
#exponential decay
#k = 0.5
decay = np.zeros((X_train.shape[0], X_train.shape[1]/30))
decay[:,:] = np.array(range(24))[::-1]
decay = k**decay
decay = decay * mask_train

#set up validation time steps
mask_valid[:,:] = 0
mask_valid[:,-1] = 1


model.fit([X_train, aux_s_train, aux_d_train, m_y_train], y_train,\
          batch_size=128, epochs=500, shuffle=True, \
          validation_data = ([X_valid, aux_s_valid, aux_d_valid, m_y_valid], y_valid, mask_valid), \
          initial_epoch = 0, verbose=2, callbacks=[checkpointer,earlystopper], sample_weight = decay)
#**********************



#****model evaluation****
model = load_model('y_mconv_lstm.hdf5' + str(k))
prob_pred = model.predict([X_test, aux_s_test, aux_d_test, m_y_test], verbose=1, batch_size = 1000)
prob_pred = prob_pred[:,-1,0]
y_test = y_test[:,-1,0]
class_pred = (prob_pred > 0.5).astype(int)

auc_ROC = roc_auc_score(y_test, prob_pred)
mcc = matthews_corrcoef(y_test, class_pred)
prfs = precision_recall_fscore_support(y_test, class_pred)
acc = accuracy_score(y_test, class_pred)
auc_PR = average_precision_score(y_test, prob_pred, average="micro")

loss = log_loss(y_test, prob_pred)
print(tabulate([['log_loss', loss], \
                ['auc@roc', auc_ROC], \
                ['mcc', mcc], \
                ['precision', prfs[0][1]], \
                ['recall', prfs[1][1]], \
                ['f1 score', prfs[2][1]],\
                ['support', prfs[3][1]],\
                ['accuray', acc], \
                ['auc@pr', auc_PR]]))
#************************
