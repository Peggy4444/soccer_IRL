#!/usr/bin/env python
# coding: utf-8

# This script trains the cnn-lstm for cloning offensive and defensive behavior of expert teams,
# saves the gradients in the directory

import tensorflow as tf
tf.test.gpu_device_name()

import numpy
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
from sklearn.model_selection import train_test_split

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import pandas as pd
import numpy as np

from google.colab import files
uploaded = files.upload()
import os
import io


path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path

directory= 'soccer_dataset/'


defensive_possession_labels = open('defensive_actions.pkl','rb')
defensive_possession_labels = pickle.load(defensive_possession_labels, encoding='latin1')

offensive_possession_labels = open('offensive_actions.pkl','rb')
offensive_possession_labels = pickle.load(offensive_possession_labels, encoding='latin1')

all_Opponent_possessions_array = open('opponent_states.pkl','rb')
all_Opponent_possessions_array = pickle.load(all_Opponent_possessions_array, encoding='latin1')

all_Expert_possessions_array = open('expert_states.pkl','rb')
all_Expert_possessions_array = pickle.load(all_Expert_possessions_array, encoding='latin1')



# Offensive network:




X_train, X_test, y_train, y_test = train_test_split(all_Expert_possessions_array, offensive_possession_labels_array)


X_train = sequence.pad_sequences(X_train, maxlen=10)
X_test = sequence.pad_sequences(X_test, maxlen=10)


from keras.utils import to_categorical
y_train= to_categorical(y_train)
y_test= to_categorical(y_test)


get_ipython().run_cell_magic('time', '', 'from keras.layers.convolutional import Conv1D \nimport matplotlib.pyplot as plt\n\n#from keras.layers import Conv1D, MaxPoling1D\noff_model = Sequential()\noff_model.add(Conv1D(filters=32, kernel_size=3, activation=\'relu\', input_shape=(10,36)))\nmodel.add(MaxPooling1D(pool_size=1))\noff_model.add(LSTM(100))\noff_model.add(Dense(2, activation=\'sigmoid\'))\noff_model.compile(loss=\'categorical_crossentropy\', optimizer=\'adam\', metrics=[\'accuracy\'])\nprint(off_model.summary())\noff_history= off_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)\n# Final evaluation of the model\nscores = off_model.evaluate(X_test, y_test, verbose=0)\nprint(off_model.summary())\n# Final evaluation of the model\nprint("accuracy: %.2f%%" % (scores[1]*100))\nprint(off_history.history.keys())\n# Final evaluation of the model\n#scores = def_model.evaluate(X_test1, y_test1, verbose=0)\nprint(off_model.summary())\n# Final evaluation of the model\n#print("accuracy: %.2f%%" % (scores[1]*100))\n\n# summarize history for accuracy\nplt.plot(off_history.history[\'accuracy\'])\nplt.plot(off_history.history[\'val_accuracy\'])\nplt.title(\'model accuracy\')\nplt.ylabel(\'accuracy\')\nplt.xlabel(\'epoch\')\nplt.legend([\'train\', \'test\'], loc=\'upper left\')\nplt.show()\n# summarize history for loss\nplt.plot(off_history.history[\'loss\'])\nplt.plot(off_history.history[\'val_loss\'])\nplt.title(\'model loss\')\nplt.ylabel(\'loss\')\nplt.xlabel(\'epoch\')\nplt.legend([\'train\', \'test\'], loc=\'upper left\')\nplt.show()')


# evaluate the model
_, train_acc = off_model.evaluate(X_train, y_train, verbose=0)
_, test_acc = off_model.evaluate(X_test, y_test, verbose=0)

# predict probabilities for test set
off_probs = off_model.predict(X_test, verbose=0)
# predict crisp classes for test set
offhat_classes = off_model.predict_classes(X_test, verbose=0)




# Defensive network:



X_train1, X_test1, y_train1, y_test1 = train_test_split(all_Apponent_possessions_array, defensive_possession_labels_array)


X_train1 = sequence.pad_sequences(X_train1, maxlen=10)
X_test1 = sequence.pad_sequences(X_test1, maxlen=10)

from keras.utils import to_categorical
y_train1= to_categorical(y_train1)
y_test1= to_categorical(y_test1)


!mkdir -p saved_model

get_ipython().system('mkdir -p saved_model')
def_model.save('saved_model/def_model')
off_model.save('saved_model/off_model')


# export weights



for layer in off_model.layers: print(layer.get_config(), layer.get_weights())



def all_weights(model):
    for layer in model.layers: 
        weight= layer.get_weights()
        print(weight)




off_layer_weights = off_model.layers[2].get_weights()[0]
def_layer_weights = def_model.layers[2].get_weights()[0]



# Saving the weights



pickle.dump(off_layer_weights, open(directory + 'off_layer_weights.pkl', 'wb'))
pickle.dump(def_layer_weights, open(directory + 'def_layer_weights.pkl', 'wb'))

