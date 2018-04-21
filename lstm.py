import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import pandas as pd
import keras
from keras.models import *
from keras.layers import *
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras import backend as K
K.set_session(session)

DATASET_ROOT = '/hdd/dataset/tbrain/'
TRAIN_PATH = os.path.join(DATASET_ROOT, 'taetfp_utf8.csv')
TS_IN=260
TS_OUT=5

train_csv = pd.read_csv(TRAIN_PATH)
train_csv.sort_values(by=['日期'], ascending=True, inplace=True)

codes = list(sorted(set(train_csv['代碼'])))
fields = ['開盤價(元)', '最高價(元)', '最低價(元)', '收盤價(元)', '成交張數(張)']

split_train = []
for code in codes:
    row = train_csv.loc[train_csv['代碼']==code][fields]
    row = row.applymap(lambda x: float(x.replace(',','')) if type(x)==str else x)
    split_train.append(np.array(row))

from keras.regularizers import *

# custom R2-score metrics for keras backend
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def make_model(ts_in=60, ts_out=5, n_field=5, n_encode=128, n_decode=128, reg_a=0.1):
    input_ = Input(shape=(ts_in, n_field))
    lstm_1 = CuDNNLSTM(n_encode, return_sequences=False, recurrent_regularizer=l2(reg_a)) (input_)
    rep_vec_2 = RepeatVector(ts_out) (lstm_1)
    lstm_3 = CuDNNLSTM(n_decode, return_sequences=True, recurrent_regularizer=l2(reg_a)) (rep_vec_2)
    lstm_4 = CuDNNLSTM(n_decode, return_sequences=True, recurrent_regularizer=l2(reg_a)) (lstm_3)
    fc_4 = TimeDistributed(Dense(1, kernel_regularizer=l2(reg_a))) (lstm_4)
    return Model([input_], [fc_4])
make_model(ts_in=TS_IN, ts_out=TS_OUT).summary()

def make_seq(seq, ts_in=60, ts_out=5):
    Xs = []
    Ys = []
    total_l = ts_in + ts_out
    for i in range(0, len(seq)-total_l+1):
        Xs.append(seq[i:i+ts_in])
        Ys.append(seq[i+ts_in:i+total_l, 3:4])
    return np.array(Xs), np.array(Ys)

train_18 = []
for seq in split_train:
    Xs, Ys = make_seq(seq, ts_in=TS_IN, ts_out=TS_OUT)
    train_18.append([Xs, Ys])

from keras.callbacks import *
from sklearn.model_selection import train_test_split
BATCH_SIZE = 64
EPOCH = 500
for model_n, (X, Y) in enumerate(train_18):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    prefix = str(model_n)+'_'
    model = make_model(ts_in=TS_IN, ts_out=TS_OUT)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[r2_keras])
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test) , batch_size=BATCH_SIZE, epochs=EPOCH, callbacks=[TensorBoard(log_dir=prefix+'logs')], shuffle=True)
    model.save(prefix+'model.h5')
    K.clear_session()


