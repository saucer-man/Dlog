import numpy as np
import keras
import os
from lib.common import save, load
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error


def LSTM_model(trainx, trainy):
    # use the train
    model = Sequential()
    model.add(LSTM(100, activation = 'relu', return_sequences = True, input_shape=(trainx.shape[1], trainx.shape[2])))
    model.add(LSTM(100, activation = 'relu'))
    model.add(Dense(trainx.shape[2]))
    model.compile(loss='mse', optimizer='adam')
    # model.fit(trainx, trainy, epochs = 50, verbose=2, callbacks=[callbacks])
    model.fit(trainx, trainy, epochs=50, verbose=2)
    model.summary()
    return model