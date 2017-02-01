from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


def get_simple_net(dictionary_length, max_length):
    '''
    Простенькая LSTM модель
    '''

    model = Sequential()
    model.add(Embedding(dictionary_length + 1, 32, input_length=max_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def get_dropout_net(dictionary_length, max_length):
    '''
    LSTM модель с dropout слоями (решение проблемы переобучения)
    '''

    model = Sequential()
    model.add(Embedding(dictionary_length + 1, 32, input_length=max_length, dropout=0.2))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
