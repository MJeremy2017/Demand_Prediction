import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import sklearn.metrics as m
from sklearn.model_selection import cross_val_score, train_test_split

# the whole data set is only from 2013 to 2016

dat = pd.read_csv('D:\Sports Demand Prediction\sportsdemand\Demand prediction\datasetFInal\dtClean.csv')
dat.shape
dat.columns

# check unique leagues
dat['League'].unique()
dat['Home'].nunique()
dat['Away'].nunique()

one_hot_dat = pd.concat([pd.get_dummies(dat[col]) for col in ['Home', 'Away', 'League']], axis=1)
dat = pd.concat([one_hot_dat, dat], axis=1)
dat.drop(['Home', 'Away', 'League'], axis=1, inplace=True)

# cross validation
X_train, X_test, y_train, y_test = train_test_split(dat.drop('Stake', axis=1),
                                                    dat['Stake'], test_size=0.33)


def mape(y_pred, y_true):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def valid_mape(clf):
    clf = clf.fit(X_train, np.log(y_train))
    y_pred = clf.predict(X_test)
    acc = 100 - mape(np.exp(y_pred), y_test)
    print('Accuracy is {0}%'.format(acc))
    return acc


valid_mape(LGBMRegressor(learning_rate=.01, n_estimators=1200))  # 74%

# try spanish league
dat = pd.read_csv('D:\Sports Demand Prediction\sportsdemand\Demand prediction\datasetFInal\dtClean.csv')
dat_sp = dat[dat['League'] == 'Spanish League']

one_hot_sp = pd.concat([pd.get_dummies(dat_sp[col]) for col in ['Home', 'Away']], axis=1)
dat_sp = pd.concat([one_hot_sp, dat_sp], axis=1)
dat_sp.drop(['Home', 'Away', 'League'], axis=1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(dat_sp.drop('Stake', axis=1),
                                                    dat_sp['Stake'], test_size=0.33)

valid_mape(LGBMRegressor(learning_rate=.01, n_estimators=1200))  # 76%
