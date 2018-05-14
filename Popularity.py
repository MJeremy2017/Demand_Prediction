# get the match popularity
# test it on dtClean

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dt_clean = pd.read_csv('CleanData/dtClean.csv')
dt_clean.columns
dt_clean['Year'].describe()

dt_clean.groupby('Year', axis=0).count()

# use 2013-2015 as training data 2016 as test set
train_X = dt_clean[dt_clean['Year'].isin([2013, 2014, 2015])][['League', 'Home', 'Away', 'Stake']]
test_X = dt_clean[dt_clean['Year'] == 2016][['League', 'Home', 'Away', 'Stake']]

# League popularity defined by stake
league_stake = train_X[['League', 'Stake']].groupby(['League'], as_index=False).\
    agg({'Stake': {'Total_Stake': 'sum'}})
league_stake.columns = [league_stake.columns.droplevel(1)]

sns.barplot(x='League', y='Stake', data=league_stake)
plt.show()

sns.distplot(league_stake['Stake'])

league_stake = train_X[['League', 'Stake']].groupby(['League'], as_index=False).\
    agg({'Stake': {'avg_Stake': 'mean'}})
league_stake.columns = [league_stake.columns.droplevel(1)]

sns.distplot(league_stake['Stake'])
plt.show()

# define league popularity by avg stake

league_stake['league_pop'] = 0
cond1 = league_stake['Stake'] < np.percentile(league_stake['Stake'], 25)
cond2 = (league_stake['Stake'] >= np.percentile(league_stake['Stake'], 25)) & \
        (league_stake['Stake'] < np.percentile(league_stake['Stake'], 50))
cond3 = (league_stake['Stake'] >= np.percentile(league_stake['Stake'], 50)) & \
        (league_stake['Stake'] < np.percentile(league_stake['Stake'], 75))
cond4 = league_stake['Stake'] >= np.percentile(league_stake['Stake'], 75)

league_stake.loc[cond1, 'league_pop'] = 1
league_stake.loc[cond2, 'league_pop'] = 2
league_stake.loc[cond3, 'league_pop'] = 3
league_stake.loc[cond4, 'league_pop'] = 4

# match popularity
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
#
# an_lm = ols('np.log(Stake) ~ C(Home) + C(Away)', data=dt_clean).fit()
# sm.stats.anova_lm(an_lm, typ=2)

# team popularity: the average stake of a team appeared in a match
a = train_X[['Home', 'Stake']].rename(columns={'Home': 'teams'})
b = train_X[['Away', 'Stake']].rename(columns={'Away': 'teams'})
team_stake = pd.concat([a, b], axis=0, ignore_index=True)
team_stake.head()

team_pop = team_stake.groupby(['teams'], as_index=False).agg({'Stake': {'avg_stake': 'mean'}})
team_pop.columns = ['teams', 'avg_stake']

team_pop['team_pop'] = 0
cond1 = team_pop['avg_stake'] < np.percentile(team_pop['avg_stake'], 25)
cond2 = (team_pop['avg_stake'] >= np.percentile(team_pop['avg_stake'], 25)) & \
        (team_pop['avg_stake'] < np.percentile(team_pop['avg_stake'], 50))
cond3 = (team_pop['avg_stake'] >= np.percentile(team_pop['avg_stake'], 50)) & \
        (team_pop['avg_stake'] < np.percentile(team_pop['avg_stake'], 75))
cond4 = team_pop['avg_stake'] >= np.percentile(team_pop['avg_stake'], 75)

team_pop.loc[cond1, 'team_pop'] = 1
team_pop.loc[cond2, 'team_pop'] = 2
team_pop.loc[cond3, 'team_pop'] = 3
team_pop.loc[cond4, 'team_pop'] = 4

# merge
pop_clean = dt_clean.merge(league_stake[['League', 'league_pop']], on='League')

# merge league popularity and team popularity to the original
# dt_clean AND compare the accuray of test set with popularity and without popularity

test = dt_clean[dt_clean['Year'] == 2016]  # test set is 2016
test_pop = test.merge(league_stake[['League', 'league_pop']], on='League')
test_pop = test_pop.merge(team_pop[['teams', 'team_pop']], left_on='Home', right_on='teams')
test_pop = test_pop.merge(team_pop[['teams', 'team_pop']], left_on='Away', right_on='teams')
test_pop.head()


def get_pop(row):
    if row['team_pop_x']*row['team_pop_y'] < 9:
        return row['team_pop_x']*row['team_pop_y']
    else:
        return 9


test_pop['match_pop'] = test_pop.apply(lambda row: get_pop(row), axis=1)

test_pop.drop(['League'], axis=1)