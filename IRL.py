#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
import pickle


# In[4]:


df_features = pd.read_excel('/Users/pegah/Desktop/Bayern_features.xlsx')


# In[5]:


Bayern_actions = pd.read_excel('/Users/pegah/Desktop/Bayern_actions.xlsx')


# In[26]:


End_list_index=[]
for index, row in Bayern_actions.iterrows():
    if (Bayern_actions['short_team_name'].iloc[index] != Bayern_actions['short_team_name'].shift(-1).iloc[index]) and (Bayern_actions['short_team_name'].iloc[index] != Bayern_actions['short_team_name'].shift(-2).iloc[index]) and ((Bayern_actions['short_team_name'].iloc[index] == Bayern_actions['short_team_name'].shift(2).iloc[index]) or (Bayern_actions['short_team_name'].iloc[index] == Bayern_actions['short_team_name'].shift(1).iloc[index])) :
        End_list_index.append(index)
        print(index)


# In[609]:


len(End_list_index)


# In[28]:


End_list_index[:10]


# In[30]:


Bayern_End_list_index=[]
for i in End_list_index:
    if Bayern_actions['short_team_name'].iloc[i]== "Bayern München" :
        Bayern_End_list_index.append(i)


# In[32]:


len(Bayern_End_list_index)


# In[24]:


Bayern_actions['short_team_name'].shift(1).iloc[8]


# In[33]:


Apponent_End_list_index=[]
for i in End_list_index:
    if Bayern_actions['short_team_name'].iloc[i]!= "Bayern München" :
        Apponent_End_list_index.append(i)


# In[34]:


len(Apponent_End_list_index)


# In[35]:


End_list_index[:10]


# In[40]:


Bayern_End_list_index[:10]


# In[41]:


Apponent_End_list_index[:10]


# In[42]:


def bayern_possession_list_maker(Bayern_actions, Bayern_End_list_index, Apponent_End_list_index):
    all_possessions=[]
    for delays in range(0,len(Bayern_End_list_index)):
        df= df_features.iloc[Apponent_End_list_index[delays]+1:Bayern_End_list_index[delays]+1]
        pos= df.to_numpy()
        all_possessions.append(pos)
    return all_possessions


# In[43]:


Bayern_possessions= bayern_possession_list_maker(Bayern_actions, Bayern_End_list_index, Apponent_End_list_index)


# In[44]:


len(Bayern_possessions)


# In[45]:


def Apponent_possession_list_maker(Bayern_actions, Bayern_End_list_index, Apponent_End_list_index):
    all_possessions2=[]
    for delays in range(0,len(Bayern_End_list_index)):
        df= df_features.iloc[Bayern_End_list_index[delays]+1:Apponent_End_list_index[delays+1]+1]
        pos= df.to_numpy()
        all_possessions2.append(pos)
    return all_possessions2


# In[46]:


Apponent_possessions= Apponent_possession_list_maker(Bayern_actions, Bayern_End_list_index, Apponent_End_list_index)


# In[47]:


len(Apponent_possessions)


# In[48]:


def offensive_label_list_maker(Bayern_actions, Bayern_End_list_index):
    possessions_label_offensive=[]
    for i in Bayern_End_list_index:
        off_labels= Bayern_actions['type_name'].iloc[i]
        possessions_label_offensive.append(off_labels)
    return possessions_label_offensive


# In[49]:


offensive_possession_labels=offensive_label_list_maker(Bayern_actions, Bayern_End_list_index)


# In[50]:


len(offensive_possession_labels)


# In[51]:


def defensive_label_list_maker(Bayern_actions, Apponent_End_list_index):
    possessions_label_defensive=[]
    for i in Apponent_End_list_index:
        labels_defensive= Bayern_actions['type_name'].iloc[i]
        possessions_label_defensive.append(labels_defensive)
    return possessions_label_defensive


# In[52]:


defensive_possession_labels=defensive_label_list_maker(Bayern_actions, Apponent_End_list_index)


# In[53]:


del defensive_possession_labels[0]


# In[54]:


len(defensive_possession_labels)


# In[55]:


del defensive_possession_labels[-119:-1]


# In[56]:


del defensive_possession_labels[-1]


# In[57]:


len(defensive_possession_labels)


# In[59]:


print(len(offensive_possession_labels), len(Bayern_possessions))


# In[58]:


print(len(defensive_possession_labels), len(Apponent_possessions))


# In[62]:


Apponent_possessions


# In[63]:


off_possession_label_array= np.array(offensive_possession_labels).T
off_possession_label_array


# In[67]:


Apponent_possessions_array= np.array(Apponent_possessions)
Bayern_possessions_array= np.array(Bayern_possessions)


# In[66]:


Apponent_possessions_array.shape


# In[68]:


Bayern_possessions_array.shape


# In[72]:


pickle.dump(Apponent_possessions, open('Apponent_possessions.pkl', 'wb'))


# In[73]:


pickle.dump(Bayern_possessions, open('Bayern_possessions.pkl', 'wb'))


# In[74]:


all_possessions= Bayern_possessions + Apponent_possessions


# In[102]:


all_labels= offensive_possession_labels + defensive_possession_labels


# In[77]:


len(all_possessions)


# In[81]:


len(all_labels)


# In[106]:


for index, item in enumerate(all_labels):
    if item == 'shot':
        all_labels[index] = 1
    elif item == 'tackle':
        all_labels[index] = 2
    elif item == 'clearance':
        all_labels[index] = 3
    elif item == 'interception':
        all_labels[index] = 4
    else:
        all_labels[index] = 5
        


# In[78]:


pickle.dump(all_possessions, open('states.pkl', 'wb'))


# In[107]:


pickle.dump(all_labels, open('actions.pkl', 'wb'))


# In[83]:


states = open('states.pkl','rb')
states = pickle.load(states, encoding='latin1')


# In[84]:


states


# In[108]:


actions = open('actions.pkl','rb')
actions = pickle.load(actions, encoding='latin1')


# In[109]:


set(actions)


# In[91]:


set(actions)


# In[114]:


len(End_list_index)


# In[120]:


off_df= Bayern_actions.iloc[Bayern_End_list_index]


# In[121]:


off_df


# In[116]:


len(states)


# In[118]:


len(actions)


# In[128]:


new_Apponent_End_list_index= Apponent_End_list_index[:2639]


# In[129]:


len(new_Apponent_End_list_index)


# In[130]:


def_df= Bayern_actions.iloc[new_Apponent_End_list_index]


# In[131]:


def_df


# In[132]:


df= pd.concat([off_df, def_df], ignore_index=True)


# In[133]:


df


# In[134]:


df.columns


# In[136]:


rew_feat_df= df[['game_id', 'time_seconds', 'player_id','short_name', 'first_name', 'last_name','type_name', 'result_name', 'short_team_name', 'time_remaining', 'action_name']]


# In[135]:


df['game_id'].unique()


# # Generate goal_diff

# In[197]:


for index, row in df.iterrows():
    if row['short_team_name']=='Bayern München' and row['type_name']=='shot' and row['result_name']=='success' :
        df['Bayern_goals'].iloc[index]=1
    else:
        df['Bayern_goals'].iloc[index]=0
        


# In[198]:


df['Bayern_goals'].unique()


# In[200]:


df['Opponent_goals']=3


# In[201]:


for index, row in df.iterrows():
    if row['short_team_name']!='Bayern München' and row['type_name']=='shot' and row['result_name']=='success' :
        df['Opponent_goals'].iloc[index]=1
    else:
        df['Opponent_goals'].iloc[index]=0


# In[247]:


df['cum_sum_Bayern_goals'] = df.groupby(['game_id'])['Bayern_goals'].apply(lambda x: x.cumsum())


# In[248]:


df['cum_sum_Opponent_goals'] = df.groupby(['game_id'])['Opponent_goals'].apply(lambda x: x.cumsum())


# In[249]:


df


# In[250]:


df['cum_sum_Bayern_goals'].unique()


# In[251]:


df['cum_sum_Opponent_goals'].unique()


# In[252]:


df['goal_diff']= df['cum_sum_Bayern_goals']-df['cum_sum_Opponent_goals']


# In[253]:


df['goal_diff'].unique()


# In[264]:


bayern['first_name'].value_counts()
    


# In[262]:


bayern= df[df['short_team_name']== 'Bayern München']


# In[263]:


bayern


# # Generate players market value

# In[299]:


pwd


# In[399]:


players_mv= pd.read_excel('/Users/pegah/Desktop/players_mv.xlsx')


# In[400]:


players_mv['market_value']= pd.to_numeric(players_mv['market_value'],errors='coerce')


# In[401]:


players_mv.to_excel('playerss_mv.xlsx')


# In[402]:


players_mv['market_value'].unique()


# In[403]:


players_mv.columns


# In[404]:


player_mv_mapping = players_mv.set_index('player_id')['market_value'].to_dict()


# In[405]:


player_mv_mapping


# In[406]:


pickle.dump(player_mv_mapping, open('player_market_value.pkl', 'wb'))


# In[407]:


df['player_market_value'] = df['player_id'].map(player_mv_mapping)


# In[409]:


df


# # Generate pre_game features (H/A, rankings)

# In[267]:


data_files = {
    'events': 'https://ndownloader.figshare.com/files/14464685',  # ZIP file containing one JSON file for each competition
    'matches': 'https://ndownloader.figshare.com/files/14464622',  # ZIP file containing one JSON file for each competition
    'players': 'https://ndownloader.figshare.com/files/15073721',  # JSON file
    'teams': 'https://ndownloader.figshare.com/files/15073697'  # JSON file
}


# In[272]:


from tqdm.notebook import tqdm
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve
from pathlib import Path
from zipfile import ZipFile, is_zipfile
for url in tqdm(data_files.values()):
    url_s3 = urlopen(url).geturl()
    path = Path(urlparse(url_s3).path)
    file_name = path.name
    file_local, _ = urlretrieve(url_s3, file_name)
    if is_zipfile(file_local):
        with ZipFile(file_local) as zip_file:
            zip_file.extractall()


# In[273]:


def read_json_file(filename):
    with open(filename, 'rb') as json_file:
        return BytesIO(json_file.read()).getvalue().decode('unicode_escape')


# In[274]:


competitions = [
#    'England',
#     'France',
     'Germany',
#     'Italy',
#     'Spain',
#   'European Championship',
#     'World Cup'
]


# In[276]:


from io import BytesIO
dfs_matches = []
for competition in competitions:
    competition_name = competition.replace(' ', '_')
    file_matches = f'matches_{competition_name}.json'
    json_matches = read_json_file(file_matches)
    df_matches = pd.read_json(json_matches)
    dfs_matches.append(df_matches)
df_matches = pd.concat(dfs_matches)


# In[280]:


df_matches


# In[281]:


df_matches.to_excel('bundesliga_matches.xlsx')


# In[282]:


pwd


# In[285]:


df_matches1


# In[293]:


bayern_games = [2516739, 2516997, 2516869, 2517000, 2516874, 2516751, 2516757,
       2517016, 2517018, 2516891, 2516766, 2516897, 2517029, 2516901,
       2516779, 2517036, 2516910, 2516784, 2516794, 2516926, 2516928,
       2516802, 2516945, 2516946, 2516818, 2516820, 2516957, 2516829,
       2516964, 2516838, 2516974, 2516851, 2516982, 2516856]

Bayern_df = df_matches[df_matches['wyId'].isin(bayern_games)]
Bayern_df


# In[298]:


df.columns


# In[410]:


bayern_matches= pd.read_excel('bayern_matches_data.xlsx')


# In[412]:


bayern_matches.columns


# In[413]:


side_mapping = bayern_matches.set_index('wyId')['bayern_side'].to_dict()
bayern_rank_mapping = bayern_matches.set_index('wyId')['bayern_rank'].to_dict()
opponent_rank_mapping = bayern_matches.set_index('wyId')['opponent_rank'].to_dict()


# In[417]:


pickle.dump(side_mapping, open('Bayern_side.pkl', 'wb'))
pickle.dump(bayern_rank_mapping, open('Bayern_rank.pkl', 'wb'))
pickle.dump(opponent_rank_mapping, open('Opponent_rank.pkl', 'wb'))


# In[418]:


df['bayern_side'] = df['game_id'].map(side_mapping)


# In[419]:


df['bayern_rank'] = df['game_id'].map(bayern_rank_mapping)


# In[420]:


df['opponent_rank'] = df['game_id'].map(opponent_rank_mapping)


# In[421]:


df


# In[451]:


side_dict= {'home': 1, 'away':0}


# In[452]:


df['bayern_side_id'] = df['bayern_side'].map(side_dict)


# In[453]:


df


# In[434]:


df['bayern_rank_inverse']= 1/ df['bayern_rank']


# In[435]:


df['opponent_rank_inverse']= 1/ df['opponent_rank']


# In[539]:


df['norm_time_remaining']= df['time_remaining']/ 5400


# In[540]:


df


# # reward feature expectation

# In[541]:


reward_features_df= df[['bayern_rank_inverse','opponent_rank_inverse', 'bayern_side_id', 'goal_diff', 'norm_time_remaining', 'player_market_value' ]]


# In[542]:


reward_features_df


# In[543]:


reward_features_df.to_excel('reward_features_df.xlsx')


# In[544]:


reward_features_array= reward_features_df.to_numpy()


# In[547]:


reward_features_array.shape


# In[545]:


reward_features_array


# In[604]:


gamma=0.99
def feature_expectations(rewards, gamma):
    discount_factor_timestep = np.power(gamma * np.ones(rewards.shape[0]),
                                        range(rewards.shape[0]))
    discounted_return = discount_factor_timestep[np.newaxis, :, np.newaxis] * rewards
    reward_est_timestep = np.sum(discounted_return, axis=1)
    return reward_est_timestep


# In[605]:


feature_expectations(reward_features_array, gamma)


# In[594]:


discount_factor_timestep = np.power(gamma * np.ones(reward_features_array.shape[0]),
                                        range(reward_features_array.shape[0]))


# In[595]:


discount_factor_timestep


# In[596]:


discount_factor_timestep[np.newaxis, :, np.newaxis]


# In[598]:


discounted_return = discount_factor_timestep[np.newaxis, :, np.newaxis] * reward_features_array


# In[599]:


discounted_return


# In[600]:


discounted_return.shape


# In[606]:


reward_feature_vector = np.sum(discounted_return, axis=1)


# In[607]:


reward_feature_vector


# In[608]:


reward_feature_vector.shape


# In[ ]:





# In[468]:





# In[ ]:





# In[ ]:





# # import gradients

# In[614]:


gradients_df= pd.read_excel('/Users/pegah/Desktop/gradients_df.xlsx')


# In[615]:


gradients_df


# In[616]:


gradients_arr= gradients_df.to_numpy()


# In[617]:


gradients_arr


# In[618]:


mean_gradients = np.mean(gradients_arr, axis=0)


# In[619]:


mean_gradients


# In[ ]:





# # compute psi: feature expectation * gradients

# In[622]:


reward_feature_vector.shape


# In[626]:


mean_gradients= mean_gradients.reshape(4,1)


# In[627]:


mean_gradients


# In[629]:


psi= np.dot(mean_gradients,reward_feature_vector)


# In[630]:


psi


# # recover feature weights: omegas

# In[631]:


ns = sp.linalg.null_space(psi)


# In[632]:


ns


# In[633]:


ns.shape


# In[634]:


omega= np.mean(ns, axis=1)


# In[635]:


omega


# In[ ]:





# In[636]:


weights = ns[:, 0] / np.sum(ns[:, 0])
#loss = np.dot(np.dot(weights.T, P), weights)


# In[637]:


weights


# In[640]:


P = np.dot(psi.T, psi)
loss = np.dot(np.dot(weights.T, P), weights)


# In[641]:


loss


# In[ ]:




