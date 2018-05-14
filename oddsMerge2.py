# merge the odds data of 1X2 and 1/2 Goal to the original data
import pandas as pd
import datetime

dt_clean = pd.read_csv('CleanData/dtClean.csv')
Odds = pd.read_csv('CleanData/Odds_13-16.csv')

dt_clean.describe()
Odds.describe()

Odds['date'] = pd.to_datetime(Odds['date'])

Odds.columns
Odds['Year'] = Odds['date'].dt.year
Odds['Month'] = Odds['date'].dt.month
Odds['DOW'] = Odds['date'].dt.dayofweek

match_Odds = dt_clean.merge(Odds, left_on=['Home', 'Away', 'Year', 'Month', 'DOW'],
                            right_on=['home', 'away', 'Year', 'Month', 'DOW'])

match_Odds.drop(['date', 'league', 'home', 'away'], axis=1, inplace=True)
match_Odds.head()

match_Odds.to_csv('CleanData/match_Odds.csv', index=False)