# merge the odds data of 1X2 to the original data
import pypyodbc
import pandas as pd
import re
import predict_with_betype

conn = pypyodbc.connect(
    "Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
    "Dbq=D:\R\CustomerPrediction\Odds - 2010 to 2016 Oct.accdb;")

q = "select * from sprtOdds WHERE `Bet Type` = '1X2'"
odds = pd.read_sql(q, conn)
odds.columns
sp_odds = odds[odds['league name'] == 'SPANISH LEAGUE']
sp_odds.to_csv('sp_odds.csv', index=False)

# let sp_odds = odds
full_odds = odds
full_odds.to_csv('all_odds.csv', index=False)
# process the book title to home and away
# 111 teamA v- teamB(...) split by ' '
split_teams = [re.split(' v | - ', teams) for teams in sp_odds['book title']]
home_team = [re.sub('\d', '', teams[0]).strip() for teams in split_teams]
away_team = [teams[1] for teams in split_teams]

len(set(home_team))  # 35
len(set(away_team))  # 35
set(away_team).difference(set(home_team))

# get the open odds for home, away and draw
# notice that the selection has the home and away
# return the league, time, home, away, open_odds_home, open_odds_away, open_odds_draw


def get_odds():
    leagues = []
    times = []
    years = []
    month = []
    day = []
    homes = []
    aways = []
    ooh = []
    ooa = []
    ood = []
    num_matches = sp_odds.shape[0]/3
    for i in range(num_matches):
        start_row = i*3
        print(1)
        mat_time = sp_odds['book date'].iloc[start_row]
        leagues.append(sp_odds['league name'].iloc[start_row])
        times.append(mat_time)
        years.append(mat_time.year)
        month.append(mat_time.month)
        day.append(mat_time.day)
        ood.append(sp_odds['open odds'].iloc[start_row])
        ooh.append(sp_odds['open odds'].iloc[start_row+1])
        ooa.append(sp_odds['open odds'].iloc[start_row+2])
        homes.append(sp_odds['selections'].iloc[start_row+1])
        aways.append(sp_odds['selections'].iloc[start_row+2])

    df = pd.DataFrame({'league': leagues, 'time': times, 'year': years, 'month': month,
                       'day': day, 'home': homes, 'away': aways,
                       'ooh': ooh, 'ooa': ooa, 'ood': ood})
    return df


mat_odds = get_odds()
mat_odds.shape
mat_odds.head()
mat_odds.to_csv('sp_match_odds.csv', index=False)
# try merge with the spanish league first
sp_league_betype = pd.read_csv('sp_league_betype.csv')
sp_league_betype.columns

sp_league_betype_odds = pd.merge(sp_league_betype, mat_odds,
                                 left_on=['match_year', 'match_month', 'match_day', 'Home', 'Away'],
                                 right_on=['year', 'month', 'day', 'home', 'away'],
                                 how='left')

sp_league_betype_odds = pd.read_csv('sp_league_bty_odds.csv')
sp_league_betype_odds.shape
sp_league_betype_odds.columns
sp_league_betype_odds['odd_ratio'] = sp_league_betype_odds['ooh']/sp_league_betype_odds['ooa']

# modeling
classifiers = predict_with_betype.CLASSFIERS
model = predict_with_betype.models(sp_league_betype_odds)
model.run_models(classifiers)
