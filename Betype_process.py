import pandas as pd
import numpy as np
import operator
import re

match_betype = pd.read_csv('match_betype.csv')
print(match_betype.columns)
betypes = match_betype['sim_betype'].unique()
print('There are {} bet types, which are {}'.format(len(betypes), betypes))

match_betype['team'][2].split(' v ')[0].split()

# split team get home and away
team = match_betype['team']
team = [re.sub('\d', '', mat).strip().split(' v ') for mat in team]
match_len = [len(t) for t in team]
del_ind = np.where(np.array(match_len) != 2)[0]
print('before drop has {} rows'.format(match_betype.shape[0]))
match_betype.drop(del_ind, inplace=True)
print('after drop has {} rows'.format(match_betype.shape[0]))

# redo
team = match_betype['team']
team = [re.sub('\d', '', mat).strip().split(' v ') for mat in team]
home_team = [t[0].strip() for t in team]
away_team = [t[1].strip() for t in team]

match_betype['home_team'] = home_team
match_betype['away_team'] = away_team
match_betype.head()
# deal with time

match_time = pd.to_datetime(match_betype['time'])
match_betype['match_time'] = match_time.dt.date

# group by to get number of bet types
match_num_betype = match_betype.groupby(['home_team', 'away_team', 'match_time']).\
    sim_betype.nunique().reset_index(name='number_betypes')

match_num_betype.head(5)
np.max(match_num_betype['number_betypes'])  # max 13 betypes

# get each bet type as variable for the data set
betypes_index = {}
for i in range(len(betypes)):
    betypes_index[betypes[i]] = i

print(betypes_index)


def get_bet_vars():
    betype_mat = np.zeros((match_num_betype.shape[0], 14))
    for i in range(match_num_betype.shape[0]):
        print('processing line {}'.format(i))
        line = match_num_betype.ix[i]
        home_team, away_team, match_time = line[0], line[1], line[2]
        sim_betypes = match_betype.loc[(match_betype['home_team'] == home_team) &
                                       (match_betype['away_team'] == away_team) &
                                       (match_betype['match_time'] == match_time), 'sim_betype']
        for betype in set(sim_betypes):
            idx = betypes_index.get(betype)
            betype_mat[i, idx] = 1

    return betype_mat


bet_vars = get_bet_vars()
# convert to data frame
index = [it[0] for it in sorted(betypes_index.items(), key=operator.itemgetter(1))]
bet_var_df = pd.DataFrame(bet_vars, columns=index)

match_num_betype = pd.concat([match_num_betype, bet_var_df], axis=1)
match_num_betype.to_csv('match_num_betype.csv', index=False)