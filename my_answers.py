import string

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


def window_transform_series(series, window_size):
    input_i = range(0, len(series) - window_size)
    output_i = range(window_size, len(series))

    # containers for input/output pairs
    X = [series[i:i+window_size] for i in input_i]
    y = [series[i] for i in output_i]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


# noinspection PyPep8Naming
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1), return_sequences=False))
    model.add(Dense(1))
    return model


def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    whitespace = [' ']
    extra_keep = ['é', 'è', 'â', 'à', '-', '"', "'"]

    keep = set(punctuation)
    # keep.update(extra_keep)           # disallowed by rubrik
    # keep.update(string.ascii_letters) # rejected by udacity-pa
    # keep.update(string.digits)        # rejected by udacity-pa
    # keep.update(string.whitespace)    # rejected by udacity-pa
    keep.update(whitespace)
    keep.update(string.ascii_lowercase)

    return ''.join((t for t in text if t in keep))


def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text)-window_size, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])

    return inputs, outputs


# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars), return_sequences=False))
    model.add(Dense(num_chars, activation='softmax'))
    return model
