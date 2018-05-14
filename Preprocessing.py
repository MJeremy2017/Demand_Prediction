# load in the data and do preprocessing
import pandas as pd
import numpy as np
import datetime
import sklearn.metrics as m
import time
import matplotlib.pyplot as plt
from multiscorer import MultiScorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from workalendar.asia import Singapore
# from sklearn.preprocessing import OneHotEncoder


# load in the data set with some columns already been deleted
raw_data = pd.read_csv('D:/Sports Demand Prediction/sportsdemand/Demand prediction/datasetFInal/dataset.csv')
raw_data.columns
raw_data.head()
raw_data = raw_data.drop(['Date'], axis=1)
print(raw_data.shape)  # (30451, 9)
# Index([u'League_x', u'Home', u'Away', u'Time', u'Is_Live', u'Start_Sales',
#       u'Close_Sales', u'Date', u'Number of Bet Types', u'Stake',
#       u'Month', u'Year'])

# 'Stake' is response variable
################################
# Prep1: Dealing with date --- Year, Month, Day of Week, Close - Start
# Prep2: Take into count of holidays
# Prep3: OneHotEncoder factors
# Prep4: Events Density
# ====================================== #
SG = Singapore()


# function to split date to Year, Month, Day
# input time format 12/1/10 18:00
def split_date(col):
    n = raw_data.shape[0]
    Year = []
    Month = []
    Day = []
    Hour = []
    Minute = []
    for i in range(n):
        time = raw_data[col][i].split(' ')
        date, hour_min = time[0].split('/'), time[1].split(':')
        month, day, year = int(date[0]), int(date[1]), int(date[2])
        hour, minute = int(hour_min[0]), int(hour_min[1])
        Year.append(year)
        Month.append(month)
        Day.append(day)
        Hour.append(hour)
        Minute.append(minute)

    return Year, Month, Day, Hour, Minute


# Day of Week and is_holiday, day_before_holiday, day_after_holiday
def judge_day(year, month, day):
    if year < 10:
        year = '200'+str(year)
    else:
        year = '20'+str(year)
    date = datetime.date(int(year), month, day)
    date_before = date - datetime.timedelta(days=1)
    date_after = date + datetime.timedelta(days=1)

    week_of_day = date.isoweekday()
    is_holiday = int(not SG.is_working_day(date))
    day_before_holiday = int(not SG.is_working_day(date_before))
    day_after_holiday = int(not SG.is_working_day(date_after))

    return week_of_day, is_holiday, day_before_holiday, day_after_holiday


# get the columns of match_year, match_month, match_day_of_week, start_sale_day_of_week
# days_between_sales
# for match: is_holiday, day_before_holiday, day_after_holiday

Match_Year = []
Match_Month = []
Match_Day_of_Week = []
Match_Hour = []
Match_Minite = []
Is_Holiday = []
Day_before_Holiday = []
Day_after_Holiday = []
Start_Sale_Day_of_Week = []
Days_between_Sales = []

Match_Year, Match_Month, Match_Day, Match_Hour, Match_Minite = split_date('Time')

for i in range(raw_data.shape[0]):
    dow, ih, dbh, dah = judge_day(Match_Year[i], Match_Month[i], Match_Day[i])
    Match_Day_of_Week.append(dow)
    Is_Holiday.append(ih)
    Day_before_Holiday.append(dbh)
    Day_after_Holiday.append(dah)

# get start_sale_day_of_week and days_between_sales

# no data!!!
Start_Year, Start_Month, Start_Day = split_date('Start_Sales')[0:3]
Close_Year, Close_Month, Close_Day = split_date('Close_Sales')[0:3]

for j in range(raw_data.shape[0]):
    dow = judge_day(Start_Year[j], Start_Month[j], Start_Day[j])[0]
    delta_day = (datetime.date(Close_Year[j], Close_Month[j], Close_Day[j]) -
                 datetime.date(Start_Year[j], Start_Month[j], Start_Day[j])).days
    Start_Sale_Day_of_Week.append(dow)
    Days_between_Sales.append(delta_day)

# ================================================ #
# find density of matches that day
match_date = pd.DataFrame({'match_year': Match_Year, 'match_month': Match_Month,
                           'match_day': Match_Day})

match_density = match_date.groupby(['match_year', 'match_month', 'match_day']).\
    size().reset_index(name='match_density')

# add variables to the raw_data and merge with match_density
expand_data = raw_data
expand_data['match_year'] = Match_Year
expand_data['match_month'] = Match_Month
expand_data['match_day_of_week'] = Match_Day_of_Week
expand_data['match_day'] = Match_Day
expand_data['match_hour'] = Match_Hour
expand_data['match_minute'] = Match_Minite
expand_data['is_holiday'] = Is_Holiday
expand_data['day_before_holiday'] = Day_before_Holiday
expand_data['day_after_holiday'] = Day_after_Holiday
expand_data['start_sale_day_of_week'] = Start_Sale_Day_of_Week
expand_data['days_between_sales'] = Days_between_Sales

full_data = pd.merge(expand_data, match_density, how='inner',
                     on=['match_year', 'match_month', 'match_day'])

print(full_data.columns)

# ([u'League_x', u'Home', u'Away', u'Time', u'Is_Live', u'Start_Sales',
#       u'Close_Sales', u'Date', u'Number of Bet Types', u'Stake', u'Month',
#       u'Year', u'match_year', u'match_month', u'match_day_of_week',
#       u'match_hour', u'match_minute', u'is_holiday', u'day_before_holiday',
#       u'day_after_holiday', u'start_sale_day_of_week', u'days_between_sales',
#       u'match_day', u'count'],
#       dtype='object')
full_data.to_csv('full_data.csv', index=False)
full_data = pd.read_csv('full_data.csv')
# do one hot encoding
print(full_data['Home'].nunique())  # 1130
print(full_data['Away'].nunique())  # 863

# get_dummies
cols = ['League', 'Home', 'Away', 'Live']
one_hot_data = pd.concat([pd.get_dummies(full_data[col]) for col in cols], axis=1)
print(one_hot_data.shape)  # (26073, 2066)

full_data = pd.concat((full_data, one_hot_data), axis=1)
print(full_data.shape)
full_data = full_data.drop(['League', 'Home', 'Away', 'Time', 'Live',
                            'Date'], axis=1)
print(full_data.shape)

# ===============modeling================= #
# random forest, adaptive boosting, gradient boosting, lightgbm
# lgb_params = {}
# lgb_params['learning_rate'] = 0.01
# lgb_params['n_estimators'] = 1000
# lgb_params['max_bin'] = 10
# lgb_params['subsample'] = 0.8
# lgb_params['subsample_freq'] = 10
# lgb_params['colsample_bytree'] = 0.8
# lgb_params['min_child_samples'] = 500

classifiers = {'Gradient Boosting': GradientBoostingRegressor(),
               'Adaptive Boosting': AdaBoostRegressor(),
               'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=5)}
               #  'LightGBM': LGBMRegressor(**lgb_params)}


def model_score(classifiers):
    train_X = full_data.drop('Stake', axis=1)
    train_y = full_data['Stake']
    scorer = MultiScorer({'R-Square': (m.r2_score, {}),
                          'MSE': (m.mean_squared_error, {})})
    res_score = {}

    for name, clf in classifiers.items():
        start = time.time()
        print(name)
        cross_val_score(clf, train_X, train_y, cv=5, scoring=scorer)
        results = scorer.get_results()
        res_score[name] = results

        for metric_name in results.keys():
            average_score = np.average(results[metric_name])
            print('%s : %f' % (metric_name, average_score))

        print 'time', time.time() - start, '\n\n'

    return res_score


scores = model_score(classifiers)

# Adaptive Boosting
# MSE : 403058506533.952820
# R-Square : -3.123057
# time 655.438999891
# Random Forest
# MSE : 272301615446.754395
# R-Square : -1.766236
# time 1359.63899994
# Gradient Boosting  # best
# MSE : 226282386365.980591
# R-Square : -1.311138
# time 481.330000162

# predict using GBM measured by mape


def mape(y_pred, y_true):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


X_train, X_test, y_train, y_test = train_test_split(full_data.drop('Stake', axis=1),
                                                    full_data['Stake'], test_size=0.33)


def valid_mape(clf):
    clf = clf.fit(X_train, np.log(y_train))
    y_pred = clf.predict(X_test)
    acc = 100 - mape(np.exp(y_pred), y_test)
    print('Accuracy is {0}%'.format(acc))
    return acc


valid_mape(GradientBoostingRegressor(learning_rate=.01, n_estimators=300))  # 50%
valid_mape(LGBMRegressor(learning_rate=.01, n_estimators=1200))

plt.hist(np.log(full_data['Stake']))
plt.show()

