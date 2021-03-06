# This scripts collects soccer logs for one team as expert from Wyscout, 
# transforms to SPADL format, 
# and exports state features for each action. 

import tensorflow as tf

print(tf.__version__)

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm

!pip install matplotsoccer

!pip install tables==3.6.1

!pip install socceraction

import numpy as np
import pandas as pd  # 1.0.3
import matplotsoccer
from ipywidgets import interact_manual, fixed, widgets  # 7.5.1

from io import BytesIO

from pathlib import Path

from tqdm.notebook import tqdm

from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve

from zipfile import ZipFile, is_zipfile

import pandas as pd  # version 1.0.3

from sklearn.metrics import brier_score_loss, roc_auc_score  # version 0.22.2
from xgboost import XGBClassifier  # version 1.0.2

import socceraction.vaep.features as features
import socceraction.vaep.labels as labels

from socceraction.spadl.wyscout import convert_to_spadl
from socceraction.vaep.formula import value

pip install socceraction==0.2.0




#Download Wyscout dataset


data_files = {
    'events': 'https://ndownloader.figshare.com/files/14464685',  # ZIP file containing one JSON file for each competition
    'matches': 'https://ndownloader.figshare.com/files/14464622',  # ZIP file containing one JSON file for each competition
    'players': 'https://ndownloader.figshare.com/files/15073721',  # JSON file
    'teams': 'https://ndownloader.figshare.com/files/15073697'  # JSON file
}

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

json_teams = read_json_file('teams.json')
df_teams = pd.read_json(json_teams)






df_teams1= df_teams.to_hdf('wyscout.h5', key='teams', mode='w')

df_teams

df_teams.to_hdf('wyscout.h5', key='teams', mode='w')

json_players = read_json_file('players.json')
df_players = pd.read_json(json_players)

#df_players

df_players.to_hdf('wyscout.h5', key='players', mode='a')



# Remove comments for the interested leagues

competitions = [
#    'England',
#     'France',
     'Germany',
#     'Italy',
#     'Spain',
#   'European Championship',
#     'World Cup'
]

dfs_matches = []
for competition in competitions:
    competition_name = competition.replace(' ', '_')
    file_matches = f'matches_{competition_name}.json'
    json_matches = read_json_file(file_matches)
    df_matches = pd.read_json(json_matches)
    dfs_matches.append(df_matches)
df_matches = pd.concat(dfs_matches)

#df_matches

df_matches.to_hdf('wyscout.h5', key='matches', mode='a')

for competition in competitions:
    competition_name = competition.replace(' ', '_')
    file_events = f'events_{competition_name}.json'
    json_events = read_json_file(file_events)
    df_events = pd.read_json(json_events)
    df_events_matches = df_events.groupby('matchId', as_index=False)
    for match_id, df_events_match in df_events_matches:
        df_events_match.to_hdf('wyscout.h5', key=f'events/match_{match_id}', mode='a')

#df_events

df_teams

convert_to_spadl('wyscout.h5', 'spadl.h5')

df_games = pd.read_hdf('spadl.h5', key='games')
df_actiontypes = pd.read_hdf('spadl.h5', key='actiontypes')
df_bodyparts = pd.read_hdf('spadl.h5', key='bodyparts')
df_results = pd.read_hdf('spadl.h5', key='results')

df_teams = pd.read_hdf('spadl.h5', key='teams')
df_players = pd.read_hdf('spadl.h5', key='players')
df_games = pd.read_hdf('spadl.h5', key='games')

team_name_mapping = df_teams.set_index('team_id')['team_name'].to_dict()
df_games['home_team_name'] = df_games['home_team_id'].map(team_name_mapping)
df_games['away_team_name'] = df_games['away_team_id'].map(team_name_mapping)



# imports all Bayern M??nchen games:

expert_games= df_games[(df_games['home_team_name'] == 'FC Bayern M??nchen') | 
         (df_games['away_team_name'] == 'FC Bayern M??nchen')
        ]

expert_games.shape



expert_games_ids= list(set(expert_games['game_id']))

len(expert_games_ids)

expert_df_events=[]
for game_id in expert_games_ids:   
  with pd.HDFStore('spadl.h5') as spadlstore:
      df_actions = spadlstore[f'actions/game_{game_id}']
      df_actions = (
          df_actions.merge(spadlstore['actiontypes'], how='left')
          .merge(spadlstore['results'], how='left')
          .merge(spadlstore['bodyparts'], how='left')
          .merge(spadlstore['players'], how="left")
          .merge(spadlstore['teams'], how='left')
          .reset_index()
          .rename(columns={'index': 'action_id'})
      )
      expert_df_events.append(df_actions)
  expert_actions= pd.concat(expert_df_events)

expert_actions

expert_actions.columns

list(set(expert_actions['short_team_name']))



def nice_time(row):
    minute = int((row['period_id']>=2) * 45 + (row['period_id']>=3) * 15 + 
                 (row['period_id']==4) * 15 + row['time_seconds'] // 60)
    second = int(row['time_seconds'] % 60)
    return f'{minute}m{second}s'



expert_actions['nice_time'] = expert_actions.apply(nice_time,axis=1)

end_first_half = expert_actions[expert_actions.period_id == 1][['game_id','time_seconds']].groupby('game_id', as_index=False).max()

end_first_half

end_second_half = expert_actions[expert_actions.period_id == 2][['game_id','time_seconds']].groupby('game_id', as_index=False).max()

end_second_half

expert_actions[expert_actions.period_id== 2]

expert_actions= pd.merge(expert_actions, end_first_half[['game_id','time_seconds']].rename(columns={'time_seconds':'half_max_second'}), on='game_id')

expert_actions= pd.merge(expert_actions, end_second_half[['game_id','time_seconds']].rename(columns={'time_seconds':'2_half_max_second'}), on='game_id')

expert_actions.columns



# time remaining feature

def time_remaining(row):
    if row['period_id']==1:
        return int(row['half_max_second']) - int(row['time_seconds'])
    if row['period_id']==2:
        return int(row['2_half_max_second']) - int(row['time_seconds'])
    
expert_actions['time_remaining'] = expert_actions.apply(time_remaining,axis=1)

def action_name(row):
    return f"{row['action_id']}: {row['nice_time']} - {row['short_name']} {row['type_name']}"

expert_actions['action_name'] = expert_actions.apply(action_name, axis=1)

PITCH_LENGTH = 105
PITCH_WIDTH = 68

for side in ['start', 'end']:
    # Normalize the X location
    key_x = f'{side}_x'
    expert_actions[f'{key_x}_norm'] = expert_actions[key_x] / PITCH_LENGTH

    # Normalize the Y location
    key_y = f'{side}_y'
    expert_actions[f'{key_y}_norm'] = expert_actions[key_y] / PITCH_WIDTH

GOAL_X = PITCH_LENGTH
GOAL_Y = PITCH_WIDTH / 2

for side in ['start', 'end']:
    diff_x = GOAL_X - expert_actions[f'{side}_x']
    diff_y = abs(GOAL_Y - expert_actions[f'{side}_y'])
    expert_actions[f'{side}_distance_to_goal'] = np.sqrt(diff_x ** 2 + diff_y ** 2)
    expert_actions[f'{side}_angle_to_goal'] = np.divide(diff_x, diff_y, 
                                                    out=np.zeros_like(diff_x), 
                                                    where=(diff_y != 0))

pd.get_dummies(expert_actions['type_name'])

def add_action_type_dummies(df_actions):
    return df_actions.merge(pd.get_dummies(df_actions['type_name']), how='left',
                             left_index=True, right_index=True)

expert_actions = add_action_type_dummies(expert_actions)

def add_distance_features(df_actions):
    df_actions['diff_x'] = df_actions['end_x'] - df_actions['start_x']
    df_actions['diff_y'] = df_actions['end_y'] - df_actions['start_y']
    df_actions['distance_covered'] = np.sqrt((df_actions['end_x'] - df_actions['start_x']) ** 2 +
                                             (df_actions['end_y'] - df_actions['start_y']) ** 2)

def add_time_played(df_actions):
    df_actions['time_played'] = (df_actions['time_seconds'] + 
                             (df_actions['period_id'] >= 2) * (45 * 60) + 
                             (df_actions['period_id'] >= 3) * (15 * 60) + 
                             (df_actions['period_id'] == 4) * (15 * 60)
                             )

add_distance_features(expert_actions)
add_time_played(expert_actions)

expert_actions.shape

expert_actions.to_excel('expert_actions.xlsx')

from google.colab import files
expert_actions.to_excel('expert_actions.xlsx')
#files.download("data.csv")

files.download("expert_actions.xlsx")

expert_actions.columns



# generate state features

df_features=expert_actions[['period_id','bodypart_id','type_id', 'result_id','start_x_norm', 'start_y_norm', 'end_x_norm', 'end_y_norm',
       'start_distance_to_goal', 'start_angle_to_goal', 'end_distance_to_goal',
       'end_angle_to_goal', 'clearance', 'corner_crossed', 'corner_short', 'cross', 'dribble',
       'foul', 'freekick_crossed', 'freekick_short', 'goalkick',
       'interception', 'keeper_save', 'pass', 'shot', 'shot_freekick',
       'shot_penalty', 'tackle', 'take_on', 'throw_in', 'diff_x', 'diff_y',
       'distance_covered', 'time_played', 'time_remaining' ]]

df_features

df_features.to_excel('expert_features.xlsx')

files.download("expert_features.xlsx")
