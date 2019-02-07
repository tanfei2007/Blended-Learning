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


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from keras.layers import Input, concatenate
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

from utility import *
#**********************

feature = ['msno', 'X', 'y', 'aux_s', 'aux_d', 'mask', 'm_y']
data = read_hdf5('1_valid.hdf5', feature)
id_valid = data[0]
X_valid = data[1]
y_valid = data[2]
aux_s_valid = data[3]
aux_d_valid = data[4]
mask_valid = data[5]
m_y_valid = data[6]
print 'loaded done'
y_valid = np.concatenate((m_y_valid, y_valid), axis = 1)
m_y_valid = np.concatenate((np.zeros((m_y_valid.shape[0], 1, 1)), m_y_valid), axis = 1)



print_id_label('valid', id_valid, y_valid)

model = load_model('y_mconv_lstm.hdf50.8')

#****model evaluation on validation****
prob_pred = model.predict([X_valid, aux_s_valid, aux_d_valid, m_y_valid], verbose=1, batch_size = 10000)
prob_pred = prob_pred[:,-1,0]
y_valid = y_valid[:,-1,0]
class_pred = (prob_pred > 0.5).astype(int)

auc_ROC = roc_auc_score(y_valid, prob_pred)
mcc = matthews_corrcoef(y_valid, class_pred)
prfs = precision_recall_fscore_support(y_valid, class_pred)
acc = accuracy_score(y_valid, class_pred)
auc_PR = average_precision_score(y_valid, prob_pred, average="micro")

loss = log_loss(y_valid, prob_pred)
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


df = pd.DataFrame({'true': y_valid.flatten().tolist(), 'pred': prob_pred.flatten().tolist()},
                index = range(len(y_valid.flatten().tolist())))
df.to_csv('rslt.csv', index = False)


"""del data, id_valid, X_valid, y_valid

#on test
feature = ['msno', 'X']
data = read_hdf5('2_test.hdf5', feature)
id = data[0]
X = data[1]

prob_pred = model.predict(X, verbose=1, batch_size = 10000)"""

#adjust probability
"""b_prime_ = 0.07019010976018287
prob_pred = adjust_prob(prob_pred, b = 0.5, b_prime = b_prime_)"""

"""df = pd.DataFrame({'msno': id.flatten(), 'is_churn': prob_pred.flatten()}, index = range(prob_pred.shape[0]))
df.to_csv('submission.csv', index = False)"""
