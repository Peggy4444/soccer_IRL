# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AlGY_DW8MQwX1Sj9RoBWuuWrq-3Qs_FJ
"""

import tensorflow as tf
tf.test.gpu_device_name()

import numpy
#from keras.datasets import imdb
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

import io
df_features = pd.read_excel('Bayern_features.xlsx')
Bayern_actions = pd.read_excel('Bayern_actions.xlsx')

End_list_index=[]
for index, row in Bayern_actions.iterrows():
    if (Bayern_actions['short_team_name'].iloc[index] != Bayern_actions['short_team_name'].shift(-1).iloc[index]) and (Bayern_actions['short_team_name'].iloc[index] != Bayern_actions['short_team_name'].shift(-2).iloc[index]) and ((Bayern_actions['short_team_name'].iloc[index] == Bayern_actions['short_team_name'].shift(2).iloc[index]) or (Bayern_actions['short_team_name'].iloc[index] == Bayern_actions['short_team_name'].shift(1).iloc[index])) :
        End_list_index.append(index)

len(End_list_index)

End_list_index[:10]

Bayern_End_list_index=[]
for i in End_list_index:
    if Bayern_actions['short_team_name'].iloc[i]== "Bayern München" :
        Bayern_End_list_index.append(i)

len(Bayern_End_list_index)

Apponent_End_list_index=[]
for i in End_list_index:
    if Bayern_actions['short_team_name'].iloc[i]!= "Bayern München" :
        Apponent_End_list_index.append(i)

len(Apponent_End_list_index)

End_list_index[:10]

Bayern_End_list_index[:10]

Apponent_End_list_index[:10]

def bayern_possession_list_maker(Bayern_actions, Bayern_End_list_index, Apponent_End_list_index):
    all_possessions=[]
    for delays in range(0,len(Bayern_End_list_index)):
        df= df_features.iloc[Apponent_End_list_index[delays]+1:Bayern_End_list_index[delays]+1]
        pos= df.to_numpy()
        all_possessions.append(pos)
    return all_possessions

Bayern_possessions= bayern_possession_list_maker(Bayern_actions, Bayern_End_list_index, Apponent_End_list_index)

len(Bayern_possessions)

def Apponent_possession_list_maker(Bayern_actions, Bayern_End_list_index, Apponent_End_list_index):
    all_possessions2=[]
    for delays in range(0,len(Bayern_End_list_index)):
        df= df_features.iloc[Bayern_End_list_index[delays]+1:Apponent_End_list_index[delays+1]+1]
        pos= df.to_numpy()
        all_possessions2.append(pos)
    return all_possessions2

Apponent_possessions= Apponent_possession_list_maker(Bayern_actions, Bayern_End_list_index, Apponent_End_list_index)

len(Apponent_possessions)

Bayern_End_list_index[:10]

def offensive_label_list_maker(Bayern_actions, Bayern_End_list_index):
    possessions_label_offensive=[]
    for i in Bayern_End_list_index:
        off_labels= Bayern_actions['type_name'].iloc[i]
        possessions_label_offensive.append(off_labels)
    return possessions_label_offensive

offensive_possession_labels=offensive_label_list_maker(Bayern_actions, Bayern_End_list_index)

len(offensive_possession_labels)

def defensive_label_list_maker(Bayern_actions, Apponent_End_list_index):
    possessions_label_defensive=[]
    for i in Apponent_End_list_index:
        labels_defensive= Bayern_actions['type_name'].iloc[i]
        possessions_label_defensive.append(labels_defensive)
    return possessions_label_defensive

defensive_possession_labels=defensive_label_list_maker(Bayern_actions, Apponent_End_list_index)

del defensive_possession_labels[0]

len(defensive_possession_labels)

del defensive_possession_labels[-119:-1]

del defensive_possession_labels[-1]

print(len(offensive_possession_labels), len(Bayern_possessions))

print(len(defensive_possession_labels), len(Apponent_possessions))

off_possession_label_array= np.array(offensive_possession_labels).T
off_possession_label_array

Apponent_possessions_array= np.array(Apponent_possessions)
Bayern_possessions_array= np.array(Bayern_possessions)

all_Bayern_possessions_array= np.array(Bayern_possessions)
all_Bayern_possessions_array.shape

all_Apponent_possessions_array= np.array(Apponent_possessions)
all_Apponent_possessions_array.shape

Bayern_ending_actions = list(set(offensive_possession_labels))

Bayern_ending_actions

Apponent_ending_actions = list(set(defensive_possession_labels))

for index, item in enumerate(offensive_possession_labels):
    if item == 'shot':
        offensive_possession_labels[index] = 1
    elif item != 'shot':
        offensive_possession_labels[index] = 0

list(set(offensive_possession_labels))

for index, item in enumerate(defensive_possession_labels):
    if item == 'tackle':
        defensive_possession_labels[index] = 0
    elif item == 'clearance':
        defensive_possession_labels[index] = 1
    elif item == 'interception':
        defensive_possession_labels[index] = 2
    else:
        defensive_possession_labels[index] = 3

list(set(defensive_possession_labels))

offensive_possession_labels_array= np.array(offensive_possession_labels).T

defensive_possession_labels_array= np.array(defensive_possession_labels).T



"""Offensive network:"""

X_train, X_test, y_train, y_test = train_test_split(all_Bayern_possessions_array, offensive_possession_labels_array)

X_train = sequence.pad_sequences(X_train, maxlen=10)
X_test = sequence.pad_sequences(X_test, maxlen=10)

y_train_cm= y_train
y_test_cm= y_test

from keras.utils import to_categorical
y_train= to_categorical(y_train)
y_test= to_categorical(y_test)

print(X_train.shape,y_train.shape)

print(X_test.shape, y_test.shape)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from keras.layers.convolutional import Conv1D 
# import matplotlib.pyplot as plt
# 
# #from keras.layers import Conv1D, MaxPoling1D
# off_model = Sequential()
# off_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(10,36)))
# #model.add(MaxPooling1D(pool_size=1))
# off_model.add(LSTM(100))
# off_model.add(Dense(2, activation='sigmoid'))
# off_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(off_model.summary())
# off_history= off_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
# # Final evaluation of the model
# scores = off_model.evaluate(X_test, y_test, verbose=0)
# print(off_model.summary())
# # Final evaluation of the model
# print("accuracy: %.2f%%" % (scores[1]*100))
# print(off_history.history.keys())
# # Final evaluation of the model
# #scores = def_model.evaluate(X_test1, y_test1, verbose=0)
# print(off_model.summary())
# # Final evaluation of the model
# #print("accuracy: %.2f%%" % (scores[1]*100))
# 
# # summarize history for accuracy
# plt.plot(off_history.history['accuracy'])
# plt.plot(off_history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(off_history.history['loss'])
# plt.plot(off_history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# evaluate the model
_, train_acc = off_model.evaluate(X_train, y_train, verbose=0)
_, test_acc = off_model.evaluate(X_test, y_test, verbose=0)

print(train_acc, test_acc)

# predict probabilities for test set
off_probs = off_model.predict(X_test, verbose=0)
# predict crisp classes for test set
offhat_classes = off_model.predict_classes(X_test, verbose=0)

offhat_classes.shape

off_probs = off_probs.reshape(660,1)

offhat_classes = offhat_classes.reshape(660,1)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test_cm, offhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test_cm, offhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test_cm, offhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test_cm, offhat_classes)
print('F1 score: %f' % f1)

set(y_test_cm) - set(offhat_classes)

from sklearn import metrics
metrics.f1_score(y_test_cm, offhat_classes, average='weighted', labels=np.unique(offhat_classes))

metrics.precision_score(y_test_cm, offhat_classes, average='weighted', labels=np.unique(offhat_classes))

metrics.recall_score(y_test_cm, offhat_classes, average='weighted', labels=np.unique(offhat_classes))

metrics.cohen_kappa_score(y_test_cm, offhat_classes, labels=np.unique(offhat_classes))

kappa = cohen_kappa_score(y_test_cm, offhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y_test_cm, offhat_classes)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_test_cm, offhat_classes)
print(matrix)







"""Defensive network:"""

X_train1, X_test1, y_train1, y_test1 = train_test_split(all_Apponent_possessions_array, defensive_possession_labels_array)

X_train1 = sequence.pad_sequences(X_train1, maxlen=10)
X_test1 = sequence.pad_sequences(X_test1, maxlen=10)

from keras.utils import to_categorical
y_train1= to_categorical(y_train1)
y_test1= to_categorical(y_test1)

print(X_train1.shape, y_train1.shape)

print(X_test1.shape, y_test1.shape)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from keras.layers.convolutional import Conv1D 
# import matplotlib.pyplot as plt
# 
# #from keras.layers import Conv1D, MaxPoling1D
# def_model = Sequential()
# def_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(10,36)))
# #model.add(MaxPooling1D(pool_size=1))
# def_model.add(LSTM(100))
# def_model.add(Dense(4, activation='softmax'))
# def_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(def_model.summary())
# def_history= def_model.fit(X_train1, y_train1, validation_data=(X_test1, y_test1), epochs=10, batch_size=64)
# print(def_history.history.keys())
# # Final evaluation of the model
# scores = def_model.evaluate(X_test1, y_test1, verbose=0)
# print(def_model.summary())
# # Final evaluation of the model
# print("accuracy: %.2f%%" % (scores[1]*100))
# 
# # summarize history for accuracy
# plt.plot(def_history.history['accuracy'])
# plt.plot(def_history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(def_history.history['loss'])
# plt.plot(def_history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

!mkdir -p saved_model
def_model.save('saved_model/def_model')

from sklearn.metrics import roc_curve
off_lable_pred = off_model.predict(X_test).ravel()
#fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, off_lable_pred)

off_model.save('saved_model/off_model')

!ls saved_model

new_model = tf.keras.models.load_model('saved_model/off_model')

# reshaping the array from 3D
# matrice to 2D matrice.
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
y_train_reshaped = y_train.reshape(y_train.shape[0], -1)
y_test_reshaped = y_test.reshape(y_test.shape[0], -1)
X_train1_reshaped = X_train1.reshape(X_train1.shape[0], -1)
X_test1_reshaped = X_test1.reshape(X_test1.shape[0], -1)
y_train1_reshaped = y_train1.reshape(y_train1.shape[0], -1)
y_test1_reshaped = y_test1.reshape(y_test1.shape[0], -1)

# saving reshaped array to file.
np.savetxt("X_train.csv", X_train_reshaped)
np.savetxt("X_test.csv", X_test_reshaped)
np.savetxt("y_train.csv", y_train_reshaped)
np.savetxt("y_test.csv", y_test_reshaped)
np.savetxt("X_train1.csv", X_train1_reshaped)
np.savetxt("X_test1.csv", X_test1_reshaped)
np.savetxt("y_train1.csv", y_train1_reshaped)
np.savetxt("y_test1.csv", y_test1_reshaped)

for layer in off_model.layers: print(layer.get_config(), layer.get_weights())

for layer in off_model.layers: 
  print(layer)

for layer in off_model.layers: 
  config=layer.get_config()
  print(config)

for layer in off_model.layers: 
  weight= layer.get_weights()
  print(weight)

weight[1]

off_layer_weights = off_model.layers[2].get_weights()[0]

print(off_layer_weights#,second_layer_biases
      )

len(off_layer_weights)

def_layer_weights = def_model.layers[2].get_weights()[0]

def_layer_weights

len(def_layer_weights)

def_mean_gradients = np.mean(def_layer_weights, axis=0)
#ns = scipy.linalg.null_space(mean_gradients)
#P = np.dot(mean_gradients.T, mean_gradients)

def_mean_gradients

off_mean_gradients = np.mean(off_layer_weights, axis=0)

off_mean_gradients

Z.T.dot(Z)

def_layer_weights.shape

from pandas import DataFrame

off_weight_df1 = DataFrame (off_layer_weights,columns=(['Shot'], ['Not Shot']))

off_weight_df1

def_weight_df1 = DataFrame (def_layer_weights,columns=(['Tackle'], ['Clearance'], ['Interception'], ['Others']))

def_weight_df1

from google.colab import drive
drive.mount('drive')

def_weight_df1.to_excel('def_weight_df1.xlsx')
!cp def_weight_df1.xlsx "drive/My Drive/"

off_weight_df1.to_excel('off_weight_df1.xlsx')
!cp off_weight_df1.xlsx "drive/My Drive/"
