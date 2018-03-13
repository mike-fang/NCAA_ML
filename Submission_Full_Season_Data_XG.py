
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt


#get_ipython().magic(u'matplotlib inline')

full_result_season = pd.read_csv('./data/DataFiles/RegularSeasonDetailedResults.csv')

#import team IDs
team_ids = pd.read_csv('./data/DataFiles/Teams.csv')

submit_sample = pd.read_csv('./data/SampleSubmissionStage1.csv')


# In[9]:


full_result_season.columns


# In[10]:


named_test = pd.merge(full_result_season, team_ids[['TeamID', 'TeamName']], left_on='WTeamID', right_on='TeamID')


# In[11]:


named_test.columns = [u'Season', u'DayNum', u'WTeamID', u'WScore', u'LTeamID', u'LScore',
       u'WLoc', u'NumOT', u'WFGM', u'WFGA', u'WFGM3', u'WFGA3', u'WFTM',
       u'WFTA', u'WOR', u'WDR', u'WAst', u'WTO', u'WStl', u'WBlk', u'WPF',
       u'LFGM', u'LFGA', u'LFGM3', u'LFGA3', u'LFTM', u'LFTA', u'LOR', u'LDR',
       u'LAst', u'LTO', u'LStl', u'LBlk', u'LPF', u'TeamID', u'WTeamName']

named_test = named_test.drop('TeamID', axis=1)


# In[12]:


named_test = pd.merge(named_test, team_ids[['TeamID', 'TeamName']], left_on='LTeamID', right_on='TeamID')


# In[13]:


named_test.columns = [u'Season',    u'DayNum',   u'WTeamID',    u'WScore',   u'LTeamID',
          u'LScore',      u'WLoc',     u'NumOT',      u'WFGM',      u'WFGA',
           u'WFGM3',     u'WFGA3',      u'WFTM',      u'WFTA',       u'WOR',
             u'WDR',      u'WAst',       u'WTO',      u'WStl',      u'WBlk',
             u'WPF',      u'LFGM',      u'LFGA',     u'LFGM3',     u'LFGA3',
            u'LFTM',      u'LFTA',       u'LOR',       u'LDR',      u'LAst',
             u'LTO',      u'LStl',      u'LBlk',       u'LPF', u'WTeamName',
          u'TeamID',  u'LTeamName']
named_test = named_test.drop('TeamID', axis=1)


# In[14]:


named_test['LDR'].plot.hist(bins = 50)


# Aggregate averages per season

# In[15]:


#Calculator Function for Stats

def point_stats(win_team_seas, loss_team_seas):

    #Points
    ppg_seas = (win_team_seas['WScore'].mean()*len_win + loss_team_seas['LScore'].mean()*len_loss)/(len_win+len_loss) 
    ppga_seas = (win_team_seas['LScore'].mean()*len_win + loss_team_seas['WScore'].mean()*len_loss)/(len_win+len_loss)

    return ppg_seas, ppga_seas
        
def rebound_stats(win_team_seas, loss_team_seas):
    # Rebounds - Defensive and Offensive
    #Defensive
    drb_seas = (win_team_seas['WDR'].mean()*len_win + loss_team_seas['LDR'].mean()*len_loss)/(len_win+len_loss)
    drba_seas = (win_team_seas['LDR'].mean()*len_win + loss_team_seas['WDR'].mean()*len_loss)/(len_win+len_loss)
    #Offensive
    orb_seas = (win_team_seas['WOR'].mean()*len_win + loss_team_seas['LOR'].mean()*len_loss)/(len_win+len_loss)
    orba_seas = (win_team_seas['LOR'].mean()*len_win + loss_team_seas['WOR'].mean()*len_loss)/(len_win+len_loss)

    return drb_seas, drba_seas, orb_seas, orba_seas
        
        
def shooting_stats(win_team_seas, loss_team_seas):
    ###Shooting Percentages
    # FG Percent
    fg_percent = (((win_team_seas['WFGM']/win_team_seas['WFGA'])*len_win).mean() 
                  + ((loss_team_seas['LFGM']/loss_team_seas['LFGA'])*len_win).mean())/(len_win+len_loss)
    # FG Percent Allowed
    fg_percent_allow = (((win_team_seas['LFGM']/win_team_seas['LFGA'])*len_win).mean() 
                  + ((loss_team_seas['WFGM']/loss_team_seas['WFGA'])*len_win).mean())/(len_win+len_loss)

    # FG 3 Pointer Percent
    fg3_percent = (((win_team_seas['WFGM3']/win_team_seas['WFGA3'])*len_win).mean() 
                  + ((loss_team_seas['LFGM3']/loss_team_seas['LFGA3'])*len_win).mean())/(len_win+len_loss)
    # FG 3 Pointer Percent Allowed
    fg3_percent_allow = (((win_team_seas['LFGM3']/win_team_seas['LFGA3'])*len_win).mean() 
                  + ((loss_team_seas['WFGM3']/loss_team_seas['WFGA3'])*len_win).mean())/(len_win+len_loss)

    #FT Percentage
    ft_percent = (((win_team_seas['WFTM']/win_team_seas['WFTA'])*len_win).mean() 
                  + ((loss_team_seas['LFTM']/loss_team_seas['LFTA'])*len_win).mean())/(len_win+len_loss)

    #FT Percentage Opponent
    ft_percent_opp = (((win_team_seas['LFTM']/win_team_seas['LFTA'])*len_win).mean() 
                  + ((loss_team_seas['WFTM']/loss_team_seas['WFTA'])*len_win).mean())/(len_win+len_loss)

    return fg_percent, fg_percent_allow, fg3_percent, fg3_percent_allow, ft_percent, ft_percent_opp

    
def assist_stats(win_team_seas, loss_team_seas):
    ###Supplementary Stats
    #Assists
    ast_seas = (win_team_seas['WAst'].mean()*len_win + loss_team_seas['LAst'].mean()*len_loss)/(len_win+len_loss) 
    asta_seas = (win_team_seas['LAst'].mean()*len_win + loss_team_seas['WAst'].mean()*len_loss)/(len_win+len_loss)

    return ast_seas, asta_seas

def turnover_stats(win_team_seas, loss_team_seas):
    #Turnovers
    to_seas = (win_team_seas['WTO'].mean()*len_win + loss_team_seas['LTO'].mean()*len_loss)/(len_win+len_loss) 
    tof_seas = (win_team_seas['LTO'].mean()*len_win + loss_team_seas['WTO'].mean()*len_loss)/(len_win+len_loss)

    return to_seas, tof_seas

def steals_stats(win_team_seas, loss_team_seas):
    #Steals
    stl_seas = (win_team_seas['WStl'].mean()*len_win + loss_team_seas['LStl'].mean()*len_loss)/(len_win+len_loss) 
    stla_seas = (win_team_seas['LStl'].mean()*len_win + loss_team_seas['WStl'].mean()*len_loss)/(len_win+len_loss)

    return stl_seas, stla_seas


def block_stats(win_team_seas, loss_team_seas):       
    #Blocks
    block_seas = (win_team_seas['WBlk'].mean()*len_win + loss_team_seas['LBlk'].mean()*len_loss)/(len_win+len_loss) 
    blocka_seas = (win_team_seas['LBlk'].mean()*len_win + loss_team_seas['WBlk'].mean()*len_loss)/(len_win+len_loss)

    return block_seas, blocka_seas
        
def foul_stats(win_team_seas, loss_team_seas):
    #Personal Fouls
    pf_seas = (win_team_seas['WPF'].mean()*len_win + loss_team_seas['LPF'].mean()*len_loss)/(len_win+len_loss) 
    pfr_seas = (win_team_seas['LPF'].mean()*len_win + loss_team_seas['WPF'].mean()*len_loss)/(len_win+len_loss)
    
    return pf_seas, pfr_seas
        
        
        
            


# In[16]:


#General Process
# 1 - Loops through WTeamIDS
# 2 - Loop through Seasons
# 3 - Calculate the desired averages
# 4 - Store in Master List

master_list = []
header = ['TeamID', 'TeamName', 'Season', 'Number Wins', 'Number Losses', 'PPG', 'PPGA', 'DRB', 'DRB Allowed', 'ORB',
          'ORB Allowed', 'FG Percent', 'FG Percent Allowed', '3 Point Percent', '3 Point Percent Alllowed', 'FT Percent',
          'FT Opponent','Assists', 'Assists Allowed' ,'Turnovers', 'Turnovers Forced', 'Steals', 'Steals Allowed', 
          'Blocks', 'Blocks Allowed', 'Personal Fouls', 'Personal Fouls Received']

#find unique teamids
unique_team_id = team_ids['TeamID'].unique()
unique_season = named_test['Season'].unique()

#Loop Through TeamIDs
for ids in unique_team_id:
    
    win_team = named_test[named_test['WTeamID']==ids]
    loss_team = named_test[named_test['LTeamID']==ids]
    
    if win_team.empty==False:
    
        team_name = win_team['WTeamName'].iloc[0]

        for seas in unique_season:
            win_team_seas = win_team[win_team['Season']==seas]
            loss_team_seas = loss_team[loss_team['Season']==seas]

            len_win = len(win_team_seas)
            len_loss = len(loss_team_seas)

            if (len_win + len_loss) != 0:

                #Call calculator functions
                #points
                ppg, ppga = point_stats(win_team_seas, loss_team_seas)
                #rebounds
                drb_seas, drba_seas, orb_seas, orba_seas = rebound_stats(win_team_seas, loss_team_seas)
                #shooting
                fg_percent, fg_percent_allow, fg3_percent, fg3_percent_allow, ft_percent, ft_percent_opp = shooting_stats(win_team_seas, loss_team_seas)
                #Assists
                ast_seas, asta_seas = assist_stats(win_team_seas, loss_team_seas)
                #Turnovers
                to_seas, tof_seas = turnover_stats(win_team_seas, loss_team_seas)
                #steals
                stl_seas, stla_seas = steals_stats(win_team_seas, loss_team_seas)
                #blocks
                block_seas, blocka_seas = block_stats(win_team_seas, loss_team_seas)
                #fouls
                pf_seas, pfr_seas = foul_stats(win_team_seas, loss_team_seas)

            else:
                ppg = 0.0
                ppga = 0.0
                drb_seas = 0.0
                drba_seas = 0.0
                orb_seas = 0.0
                orba_seas = 0.0
                fg_percent = 0.0
                fg_percent_allow = 0.0
                fg3_percent = 0.0
                fg3_percent_allow = 0.0
                ft_percent = 0.0
                ft_percent_opp = 0.0
                ast_seas = 0.0
                asta_seas = 0.0
                to_seas = 0.0
                tof_seas = 0.0
                stl_seas = 0.0
                stla_seas = 0.0
                block_seas = 0.0
                blocka_seas = 0.0
                pf_seas = 0.0
                pfr_seas = 0.0


            #storage tuple
            stor = [ids, team_name, seas, len_win, len_loss, ppg, ppga, drb_seas, drba_seas, orb_seas, orba_seas,
                   fg_percent, fg_percent_allow, fg3_percent, fg3_percent_allow, ft_percent, ft_percent_opp, 
                   ast_seas, asta_seas, to_seas, tof_seas, stl_seas, stla_seas, block_seas, blocka_seas, pf_seas, pfr_seas]
            master_list.append(stor)
            
            
            
            
            


# In[17]:


df1 = pd.DataFrame(master_list, columns = header)


# In[18]:


duke = df1[df1['TeamName'] == 'Duke']
duke = duke.sort_values(by =['Season'])


# In[19]:


duke.plot(x ='Season', y = ['PPG', 'PPGA'], color = ['skyblue', 'olive'])


# In[20]:


duke = duke.sort_values(by =['Season'])


# Manipulate data table to perform XG Boost

# function to get wins

# In[21]:


def get_team1_win(row):
    if row['WTeamID'] < row['LTeamID']:
        return 1
    else:
        return 0


# In[22]:


full_result_season['Team1_win'] = full_result_season[['Season', 'WTeamID', 'LTeamID']].apply(get_team1_win, axis=1)


# In[23]:


full_result_season['Team1'] = full_result_season[['WTeamID', 'LTeamID']].min(axis=1)
full_result_season['Team2'] = full_result_season[['WTeamID', 'LTeamID']].max(axis=1)


# Create new data frame for season stats

# In[24]:


season_stats = full_result_season[['Season', 'Team1', 'Team2', 'Team1_win']]
#Merge the desired team id and wins for Team 1
season_stats = pd.merge(season_stats, df1, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
del season_stats['TeamID']
#Merge the desired team id and wins for Team 2
season_stats = pd.merge(season_stats, df1, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left', suffixes=['_1', '_2'])
del season_stats['TeamID']


# In[25]:


season_stats.columns


# Start the modeling process

# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import xgboost as xgb


# Define the datasets

# In[27]:


no_reg_cols = ['Team1_win', 'TeamName_1', 'TeamName_2', 'Season', 'Team1', 'Team2', 'Number Losses_1', 'Number Losses_2',              'PPGA_1', 'PPGA_2']

train, val = train_test_split(season_stats, test_size=.2)

regressors = [col for col in train.columns if (col not in no_reg_cols) and ('Allowed' not in col) and ('Forced' not in col) and ('Received' not in col) and ('Opponent' not in col) and (('Alllowed' not in col))]
print('Features to be used in training: ', regressors)

X_train = train[regressors].values
y_train = train['Team1_win']

X_val = val[regressors].values
y_val = val['Team1_win']


# In[ ]:


d_train = xgb.DMatrix(X_train, label=y_train)
d_val = xgb.DMatrix(X_val, label=y_val)

print(X_train.shape)
print(X_val.shape)


# Run the xgboost model

# In[ ]:


xgb.DMatrix(X_val)

