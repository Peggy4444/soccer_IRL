#!/usr/bin/env python
# coding: utf-8

# This scripts export states and actions, and prepares the array of inputs to the neural networks for behavioral cloning
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
import os
import io


path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path

directory= 'soccer_dataset/'

df_state_features = pd.read_excel(directory +'expert_team_features.xlsx')
expert_actions = pd.read_excel(directory +'expert_actions.xlsx')

# extracting possession ending indexes: possession can be transferred if and only if the team is not in
#the possession of the ball over two consecutive events. Thus, the unsuccessful
#touches of the opponent in fewer than 3 consecutive actions are not considered
#as a possession loss.

def End_list_index_maker(expert_actions):
    End_list_index=[]
    for index, row in expert_actions.iterrows():
        if (expert_actions['short_team_name'].iloc[index] != expert_actions['short_team_name'].shift(-1).iloc[index]) and (expert_actions['short_team_name'].iloc[index] != expert_actions['short_team_name'].shift(-2).iloc[index]) and ((expert_actions['short_team_name'].iloc[index] == expert_actions['short_team_name'].shift(2).iloc[index]) or (expert_actions['short_team_name'].iloc[index] == expert_actions['short_team_name'].shift(1).iloc[index])) :
            End_list_index.append(index)
    return (End_list_index)


def Expert_list_index_maker(expert_actions):
    Expert_End_list_index=[]
    for i in End_list_index:
        # change short_team_name to the team of interest 
        if Expert_actions['short_team_name'].iloc[i]== "Bayern München" :
            Expert_End_list_index.append(i)
    return (Expert_End_list_index)


def Opponent_list_index_maker(expert_actions):
    Opponent_End_list_index=[]
    for i in End_list_index:
        if expert_actions['short_team_name'].iloc[i]!= "Bayern München" :
            Opponent_End_list_index.append(i)
    return (Opponent_End_list_index)



def expert_possession_list_maker(expert_actions, Expert_End_list_index, Opponent_End_list_index):
    all_possessions=[]
    for delays in range(0,len(Expert_End_list_index)):
        df= df_state_features.iloc[Opponent_End_list_index[delays]+1:Expert_End_list_index[delays]+1]
        pos= df.to_numpy()
        all_possessions.append(pos)
    return all_possessions



def Opponent_possession_list_maker(expert_actions, Expert_End_list_index, Opponent_End_list_index):
    all_possessions2=[]
    for delays in range(0,len(Expert_End_list_index)):
        df= df_state_features.iloc[Expert_End_list_index[delays]+1:Opponent_End_list_index[delays+1]+1]
        pos= df.to_numpy()
        all_possessions2.append(pos)
    return all_possessions2


expert_possessions= expert_possession_list_maker(expert_actions, Expert_End_list_index, Opponent_End_list_index)
opponent_possessions= Opponent_possession_list_maker(expert_actions, Expert_End_list_index, Opponent_End_list_index)



# extracting ending actions: (shot for expert team possessions), (tackles, interceptions, clearances for opponent teams possessions)

def offensive_label_list_maker(expert_actions, Expert_End_list_index):
    possessions_label_offensive=[]
    for i in Expert_End_list_index:
        off_labels= expert_actions['type_name'].iloc[i]
        possessions_label_offensive.append(off_labels)
    return possessions_label_offensive


#offensive actions 


offensive_possession_labels=offensive_label_list_maker(expert_actions, Expert_End_list_index)

def defensive_label_list_maker(expert_actions, Opponent_End_list_index):
    possessions_label_defensive=[]
    for i in Opponent_End_list_index:
        labels_defensive= expert_actions['type_name'].iloc[i]
        possessions_label_defensive.append(labels_defensive)
    return possessions_label_defensive

#defensive acttions

defensive_possession_labels=defensive_label_list_maker(expert_actions, Opponent_End_list_index)


#convert to array



all_Opponent_possessions_array= np.array(Opponent_possessions)
all_Expert_possessions_array= np.array(expert_possessions)


for index, item in enumerate(offensive_possession_labels):
    if item == 'shot':
        offensive_possession_labels[index] = 1
    elif item != 'shot':
        offensive_possession_labels[index] = 0


for index, item in enumerate(defensive_possession_labels):
    if item == 'tackle':
        defensive_possession_labels[index] = 0
    elif item == 'clearance':
        defensive_possession_labels[index] = 1
    elif item == 'interception':
        defensive_possession_labels[index] = 2
    else:
        defensive_possession_labels[index] = 3
        

# saving the states and actions

pickle.dump(defensive_possession_labels, open(directory + 'defensive_actions.pkl', 'wb'))
pickle.dump(offensive_possession_labels, open(directory + 'offensive_actions.pkl', 'wb'))
pickle.dump(all_Opponent_possessions_array, open(directory + 'opponent_states.pkl', 'wb'))
pickle.dump(all_Expert_possessions_array, open(directory + 'expert_states.pkl', 'wb'))

