import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool

spanish_league = pd.read_csv('spanish_league.csv')
spanish_league.columns

drop_cols = ['Start_Sales', 'Close_Sales', 'Date', 'Month', 'Year', 'League_x', 'Time']
cat_features = ['Home', 'Away']

def mape(y_pred, y_true):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def run_model():
    league_data = spanish_league.drop(drop_cols, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(league_data.drop('Stake', axis=1),
                                                        league_data['Stake'], test_size=0.4)
    clf = CatBoostRegressor(iterations=10, learning_rate=0.01,
        depth=3, l2_leaf_reg=15,
        bagging_temperature=8,
        loss_function='MAE',
        eval_metric='MAE')
    clf = clf.fit(X_train, np.log(y_train), cat_features=[0, 1])
    y_pred = clf.predict(X_test)

    acc = mape(np.exp(y_pred), y_test)
    print 'Accuracy is {:.2f}'.format(acc)


run_model()

league_data = spanish_league.drop(drop_cols, axis=1)
X_train, X_test, y_train, y_test = train_test_split(league_data.drop('Stake', axis=1),
                                                    league_data['Stake'], test_size=0.3)

train_pool = Pool(X_train, y_train, cat_features=[0, 1])
test_pool = Pool(X_test, y_test, cat_features=[0, 1])

clf = CatBoostRegressor(iterations=10, learning_rate=.02,
        depth=3,
        loss_function='MAPE',
        eval_metric='MAPE')
clf.fit(train_pool, eval_set=test_pool, verbose=True)
y_pred = clf.predict(X_test.values)

acc = 100 - mape(y_pred, y_test)
print 'Accuracy is {:.2f}%'.format(acc)


