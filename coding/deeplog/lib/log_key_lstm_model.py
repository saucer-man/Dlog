# coding=utf-8
# /usr/bin/env python3
# description: 给定
import tensorflow as tf
from nltk.probability import FreqDist
from nltk.util import ngrams
import numpy as np
import keras
import os
from lib.common import save, load
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error


class Mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.8):
            print("Reached 80% accuracy so stopping training")
            self.model.stop_learning = True


class log_key_model:
    # karas doc: https://keras-cn.readthedocs.io/en/latest/models/model/
    def __init__(self):
        self.model = ""
        self.callbacks = Mycallback()

    def train(self, x, y):
        batch_size = 16
        # according to the article, we will stack two layers of LSTM, the model about stacked LSTM for sequence classification
        model = Sequential()
        # =============== model 1 ===================
        # input logdata shape: (batch_size, timesteps, data_dim)
        model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        # output layer with a single value prediction (1,K)
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        # to ensure the training logdata patterns remain sequential --- disable the shuffle
        # make it stateful, we add batch_size
        model.fit(x, y, epochs=500, batch_size=batch_size, verbose=2, callbacks=[self.callbacks], shuffle=False)
        # to see the summary of input and output shape
        model.summary()
        print('the accuracy for single lstm model is:', model.evaluate(x, y, batch_size=batch_size, verbose=0))
        return model

    # 返回第几个日志序列有问题
    def predcit(self, model, x_test, y_test):
        anomaly_sequence = []
        y_pred = model.predict(x_test, verbose = 0)
        yhat = []
        top_k = 3
        for i in range(y_pred.shape[0]):
            # https://blog.csdn.net/dlhlSC/article/details/88072268
            yhat.append(y_pred[i].argsort()[::-1][0:top_k])
        # print("-------------------")
        # print(y_pred.shape)  # (76, 100)
        # print(y_pred[0])
        # print(y_pred[0].shape)  # (60, 11)
        # print(type(y_pred[1]))  # <class 'numpy.ndarray'>
        # print(y_test[0].shape)  # (11,)
        # print(type(y_test[1]))  # <class 'numpy.ndarray'>
        print('the length of yhat is {}'.format(len(yhat)))
        #
        for n in range(len(yhat)):
            # 如果yhat[n]和y_test[i].argsort()[::-1][0:top_k]有共同元素，则认为正常，否则异常
            y_test_top_k_set = set(y_test[n].argsort()[::-1][0:top_k].tolist())
            yhat_top_k_set = set(yhat[n].tolist())
            # print(y_test[n])
            # print(y_test_top_k_set)
            # print(yhat[n])
            # print(yhat_top_k_set)
            if len(y_test_top_k_set.intersection(yhat_top_k_set)) > 0:
                # print("正常")
                pass
            else:
                # print("不正常")
                anomaly_sequence.append(n)
        #         eventId = 'E' + str(n)
        #         anomaly_log_key = key_name_dict[eventId]
        #         anomaly_log_keys.append(anomaly_log_key)
        #         print("log {} is possible anomaly".format(anomaly_log_key))
        # save(anomaly_log_keys, 'logdata/anomaly_log_key.pkl')
        return anomaly_sequence