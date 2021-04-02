#!/usr/bin/env python
# coding: utf-8

# This script prepares the proposed reward features dataset for computing feature expectation


import pandas as pd
import pickle
import os




defensive_possession_labels = open('defensive_actions.pkl','rb')
defensive_possession_labels = pickle.load(defensive_possession_labels, encoding='latin1')

offensive_possession_labels = open('offensive_actions.pkl','rb')
offensive_possession_labels = pickle.load(offensive_possession_labels, encoding='latin1')

all_Opponent_possessions_array = open('opponent_states.pkl','rb')
all_Opponent_possessions_array = pickle.load(all_Opponent_possessions_array, encoding='latin1')

all_Expert_possessions_array = open('expert_states.pkl','rb')
all_Expert_possessions_array = pickle.load(all_Expert_possessions_array, encoding='latin1')




path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path

directory= 'soccer_dataset/'


players_mv = open('player_market_value.pkl','rb')
players_mv = pickle.load(market_value, encoding='latin1')




# generate goal diff feature




for index, row in df_rew_features.iterrows():
    if row['short_team_name']=='Bayern München' and row['type_name']=='shot' and row['result_name']=='success' :
        df_rew_features['Bayern_goals'].iloc[index]=1
    else:
        df_rew_features['Bayern_goals'].iloc[index]=0


for index, row in df_rew_features.iterrows():
    if row['short_team_name']!='Bayern München' and row['type_name']=='shot' and row['result_name']=='success' :
        df_rew_features['Opponent_goals'].iloc[index]=1
    else:
        df_rew_features['Opponent_goals'].iloc[index]=0



def goal_diff_maker(df_rew_features):
    df_rew_features['cum_sum_Bayern_goals'] = df_rew_features.groupby(['game_id'])['Bayern_goals'].apply(lambda x: x.cumsum())
    df_rew_features['cum_sum_Opponent_goals'] = df_rew_features.groupby(['game_id'])['Opponent_goals'].apply(lambda x: x.cumsum())
    df_rew_features['goal_diff']= df_rew_features['cum_sum_Bayern_goals']-df_rew_features['cum_sum_Opponent_goals']
    return(df)


# Generate players market value


def player_intention_maker(df_rew_features):
    player_mv_mapping = players_mv.set_index('player_id')['market_value'].to_dict()
    df_rew_features['player_market_value'] = df_rew_features['player_id'].map(player_mv_mapping)
    return(df_rew_features)


# Generate pre_game features (H/A, rankings)



data_files = {
    'events': 'https://ndownloader.figshare.com/files/14464685',  # ZIP file containing one JSON file for each competition
    'matches': 'https://ndownloader.figshare.com/files/14464622',  # ZIP file containing one JSON file for each competition
    'players': 'https://ndownloader.figshare.com/files/15073721',  # JSON file
    'teams': 'https://ndownloader.figshare.com/files/15073697'  # JSON file
}



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

def read_json_file(filename):
    with open(filename, 'rb') as json_file:
        return BytesIO(json_file.read()).getvalue().decode('unicode_escape')


# In[ ]:


competitions = [
#    'England',
#     'France',
     'Germany',
#     'Italy',
#     'Spain',
#   'European Championship',
#     'World Cup'
]


from io import BytesIO
dfs_matches = []
for competition in competitions:
    competition_name = competition.replace(' ', '_')
    file_matches = f'matches_{competition_name}.json'
    json_matches = read_json_file(file_matches)
    df_matches = pd.read_json(json_matches)
    dfs_matches.append(df_matches)
df_matches = pd.concat(dfs_matches)


bayern_games = [2516739, 2516997, 2516869, 2517000, 2516874, 2516751, 2516757,
       2517016, 2517018, 2516891, 2516766, 2516897, 2517029, 2516901,
       2516779, 2517036, 2516910, 2516784, 2516794, 2516926, 2516928,
       2516802, 2516945, 2516946, 2516818, 2516820, 2516957, 2516829,
       2516964, 2516838, 2516974, 2516851, 2516982, 2516856]

Bayern_df = df_matches[df_matches['wyId'].isin(bayern_games)]
Bayern_df


side_mapping = bayern_matches.set_index('wyId')['expert_side'].to_dict()
bayern_rank_mapping = bayern_matches.set_index('wyId')['expert_rank'].to_dict()
opponent_rank_mapping = bayern_matches.set_index('wyId')['opponent_rank'].to_dict()




def pre_game_feature_maker(df_rew_features):
    side_mapping = bayern_matches.set_index('wyId')['bayern_side'].to_dict()
    expert_rank_mapping = expert_matches.set_index('wyId')['expert_rank'].to_dict()
    opponent_rank_mapping = bayern_matches.set_index('wyId')['opponent_rank'].to_dict()
    df_rew_features['expert_side'] = df_rew_features['game_id'].map(side_mapping)
    df_rew_features['expert_rank'] = df_rew_features['game_id'].map(expert_rank_mapping)
    df_rew_features['opponent_rank'] = df_rew_features['game_id'].map(opponent_rank_mapping)
    side_dict= {'home': 1, 'away':0}
    df_rew_features['expert_side_id'] = df['expert_side'].map(side_dict)
    df_rew_features['expert_rank_inverse']= 1/ df_rew_features['expert_rank']
    df_rew_features['opponent_rank_inverse']= 1/ df_rew_features['opponent_rank']
    df_rew_features['norm_time_remaining']= df_rew_features['time_remaining']/ 5400
    return(df_rew_features)


