"""Utilities for managing the training data and models of the genderml system.

"""
import numpy as np

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential

import tensorflow as tf
#  tensorflow.enable_eager_execution()
# tf.compat.v1.enable_eager_execution()

import vggish_keras


def create_raw_model(vector_length=22050, model_label=None):
    """This model takes a raw 1 second sample at 22kHz and feeds it into a
    5 layer dense sequential ML model."""
    print("Called create_raw_model")
    model = Sequential()
    #  model.add(Dense(256, input_shape=(vector_length,)))
    model.add(Input(shape=(vector_length)))
    #  model.add(Dense(256, input_shape=(vector_length, 1)))
    #  model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # one output neuron with sigmoid activation function,
    # 0 means female, 1 means male
    model.add(Dense(1, activation="sigmoid"))
    # using binary crossentropy as it's male/female classification (binary)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"],
                  optimizer="adam")
    # print summary of the model
    model.summary()
    return model


def create_4layer_model(vector_length=22050, model_label=None):
    """4 hidden dense layers from 256 units to 64."""
    print(f"Called create_4layer_model, {vector_length}")
    model = Sequential()
    model.add(Input(shape=(vector_length)))
    #  model.add(Dense(128))  #, input_shape=(vector_length)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # one output neuron with sigmoid activation function,
    # 0 means female, 1 means male
    model.add(Dense(1, activation="sigmoid"))
    # using binary crossentropy as it's male/female classification (binary)
    # model.compile(loss="binary_crossentropy", metrics=["accuracy"],
    #               optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    # # print summary of the model
    # model.summary()
    return model


def create_5layer_model(vector_length=22050, model_label=None):
    """5 hidden dense layers from 256 units to 64, not the best model, but not
    bad."""
    print("Called create_5layer_model")
    model = Sequential()
    model.add(Input(shape=(vector_length)))
    #  model.add(Dense(128))  #, input_shape=(vector_length)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # one output neuron with sigmoid activation function,
    # 0 means female, 1 means male
    model.add(Dense(1, activation="sigmoid"))
    # using binary crossentropy as it's male/female classification (binary)
    # model.compile(loss="binary_crossentropy", metrics=["accuracy"],
    #               optimizer="adam")
    # # print summary of the model
    # model.summary()
    return model


def create_6layer_model(vector_length=22050, model_label=None):
    """6 hidden dense layers from 256 units to 64."""
    print("Called create_6layer_model")
    model = Sequential()
    model.add(Input(shape=(vector_length)))
    #  model.add(Dense(128))  #, input_shape=(vector_length)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # one output neuron with sigmoid activation function,
    # 0 means female, 1 means male
    model.add(Dense(1, activation="sigmoid"))
    # using binary crossentropy as it's male/female classification (binary)
    # model.compile(loss="binary_crossentropy", metrics=["accuracy"],
    #               optimizer="adam")
    # # print summary of the model
    # model.summary()
    return model


def create_lstm_model(vector_length=128, model_label=None):
    """Create an lstm model which uses mel_spectrogram input."""
    #  input_shape = Input(shape=(vector_length,))

    model = Sequential()
    model.add(Input(shape=(vector_length,)))
    model.add(LSTM(128))
    #  , input_shape=input_shape))
    model.add(LSTM(128))
    #  model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    #  model.add(Dropout(0.3))
    model.add(Dense(10, activation="softmax"))
    #  model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    return model

def create_vgg_cnn(model_label=None):
    """Use the VGG slim network as our base which is CNN design to
    work with tensor/array of MEL spectograms sampled at 15ms intervals.
    
    The CNN allows the network to under spectral and temporal
    features, which then goes into a fully connected NN."""
    
    # Load a sample features tensor so we build to the right dimensions

    # features_file = '/mnt/fastest/jem/ml/training_data/processed/vgg22/test/vgg_features_0.npy'
    # features = np.load(features_file, allow_pickle=True)
    # model = vggish_slim.define_vggish_slim(features_tensor=features[0],training=True)
    # model = vggish_slim.define_vggish_slim(training=True)
    # return tf.identity(net, name='embedding')

    model = vggish_keras.vggish.VGGish(input_shape=(None, 96, 64), weights=None, include_top=False, model_label=model_label)
    # using binary crossentropy as it's male/female classification (binary)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"],
                  optimizer="adam")
    # print summary of the model
    # model.summary()

    return model
