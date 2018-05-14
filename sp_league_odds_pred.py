import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor

sp_league_odds = pd.read_csv('sp_league_odds.csv')
sp_league_odds.columns

CLASSFIERS = {'Gradient Boosting': GradientBoostingRegressor(),
              'Adaptive Boosting': AdaBoostRegressor(),
              'LightGBM': LGBMRegressor(n_estimators=1000, learning_rate=0.005)}


class models():

    def __init__(self, sub_league):
        self.sub_league = sub_league
        self.drop_col = ['League_x', 'Home', 'Away', 'Time',
                         'Start_Sales', 'Close_Sales', 'Date', 'Year']  # 8 columns
        self.one_hot_cols = ['Home', 'Away']

    def one_hot_encode(self):
        one_hot_data = pd.concat([pd.get_dummies(self.sub_league[col]) for col in self.one_hot_cols], axis=1)
        self.sub_league = pd.concat((self.sub_league, one_hot_data), axis=1)
        print 'Dimension after one-hot-encoding is', self.sub_league.shape

    def drop_cols(self):
        self.sub_league.drop(self.drop_col, axis=1, inplace=True)
        print 'Dimension after dropping columns is', self.sub_league.shape

    def run_models(self, classifiers):
        self.one_hot_encode()
        self.drop_cols()

        def mape(y_pred, y_true):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # split date
        X_train, X_test, y_train, y_test = train_test_split(self.sub_league.drop('Stake', axis=1),
                                                            self.sub_league['Stake'], test_size=0.33)

        for name, clf in classifiers.items():
            print 'Running', name, '...'
            clf = clf.fit(X_train, np.log(y_train))
            y_pred = clf.predict(X_test)
            acc = 100 - mape(np.exp(y_pred), y_test)
            print('Accuracy is {:.3f}%'.format(acc))


model = models(sp_league_odds)  # 65-66%
model.run_models(classifiers=CLASSFIERS)
