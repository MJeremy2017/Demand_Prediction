import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

full_data = pd.read_csv('full_data.csv')
print full_data.columns
full_data['Is_live'] = (full_data['Is_Live'] == 'Yes').astype(int)
full_data.drop('Is_Live', axis=1, inplace=True)
full_data.to_csv('full_data.csv', index=False)
# ================================== #

full_data['match_year'].value_counts()
drop_cols = ['Home', 'Away', 'Time', 'Start_Sales',
             'Close_Sales', 'Date', 'Month', 'Year', 'match_month', 'match_day_of_week',
             'match_day', 'match_hour', 'match_minute', 'start_sale_day_of_week', 'days_between_sales']
# suppose we know the number of bet types, match density

league_data = full_data.drop(drop_cols, axis=1)
print league_data.columns
# agg data
league_agg = league_data.groupby(['League_x', 'match_year'], as_index=False).agg({
    'Number of Bet Types': {'number_bet_types_mean': 'mean',
                            'number_bet_types_median': 'median',
                            'number_bet_types_max': 'max'},
    'is_holiday': {'is_holiday_sum': 'sum'},
    'day_before_holiday': {'day_before_holiday_sum': 'sum'},
    'day_after_holiday': {'day_after_holiday_sum': 'sum'},
    'match_density': {'match_density_min': 'min',
                      'match_density_median': 'median',
                      'match_density_max': 'max'},
    'Is_live': {'Is_live_sum': 'sum'},
    'Stake': {'Stake': 'sum'}
})

league_agg.columns = [league_agg.columns.droplevel(0)]

league_agg.to_csv('league_agg.csv', index=False)
# =================================== #
league_agg = pd.read_csv('league_agg.csv')
print league_agg.columns
print league_agg['Stake'].shape

league_match_count = league_data[['League_x', 'match_year']].groupby(['League_x', 'match_year']).\
                     size().reset_index(name='number of matches')

league_match_count.to_csv('league_match_count.csv', index=False)


league_agg_match = pd.merge(league_agg, league_match_count, how='inner',
                            on=['League_x', 'match_year'])
print league_agg_match.columns

league_agg_match.to_csv('league_agg_match.csv', index=False)
# build predictive model
CLASSFIERS = {'Gradient Boosting': GradientBoostingRegressor(n_estimators=200),
              'Adaptive Boosting': AdaBoostRegressor(),
              'LightGBM': LGBMRegressor(n_estimators=300, learning_rate=0.01)}

def mape(y_pred, y_true):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# predict with out league
league_agg_match.drop('League_x', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(league_agg_match.drop('Stake', axis=1),
                                                    league_agg_match['Stake'], test_size=0.33)

for name, clf in CLASSFIERS.items():
    print'Running', name, '...'
    clf = clf.fit(X_train, np.log(y_train))
    y_pred = clf.predict(X_test)
    acc = 100 - mape(np.exp(y_pred), y_test)
    print'Accuracy is {0}%'.format(acc)



