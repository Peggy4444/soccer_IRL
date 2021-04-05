#!/usr/bin/env python
# coding: utf-8



# This script requires the recovered reward weights from GIRL, computed feature expectation, and performs clustering



import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


features_index = [[0,1,2,3,4,5]]
agents = ['Bayer Leverkusen', 'Augsburg', 'Eintracht Frankfurt', 'Borussia M’gladbach', 'K¨oln', 'Werder Bremen', 'Hoffenheim', 'Hannover 96', 'Stuttgart', 'Mainz 05'
          , 'Schalke 04', 'Wolfsburg', 'Hertha BSC', 'Freiburg', 'Hamburger SV', 'RB Leipzig', 'Borussia Dortmund']
num_agents=agents.shape[0]


# getting optimal number of clusters



X = rew_weights[:num_agents,features_index[0]]
wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(2, 10), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()



# import the recovered weights from GIRL




all_weights_arr=np.array(all_weights)




num_agents = 16
num_clusters = 3
for weight in features_index[0]:
    X = rew_weights[:num_agents,weight]
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    print("Team expert:")
    print(agents[:num_agents])
    print("Assignment")
    print(kmeans.labels_)
    print()

