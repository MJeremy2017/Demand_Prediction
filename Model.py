import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# load in the data in Preprocessing
# build model on separate teams

full_data = pd.read_csv('full_data.csv')
full_data.columns

full_data['League_x'].value_counts()

# Spanish League            2131
# Italian League            2031
# E League Champ            1733
# J League                  1724
# German League             1577
# US Soccer League          1537
# English League Champ      1398
# K League                  1299
# E Premier                 1102
# International Football    1022
# English Premier            955
# UE Europe                  930

# rewrite days between sales

# focus on Spanish League

sp_league = full_data[full_data.League_x == 'Spanish League']
sp_league.Home.nunique()
sp_league.Away.nunique()  # 33 different teams; encode in the same way

# ========================== #
CLASSFIERS = {'Gradient Boosting': GradientBoostingRegressor(),
              'Adaptive Boosting': AdaBoostRegressor(),
              'LightGBM': LGBMRegressor(n_estimators=1000, learning_rate=0.005),
              'MLP': MLPRegressor(hidden_layer_sizes=(200, 300),
                                  batch_size=32,
                                  learning_rate='adaptive',
                                  learning_rate_init=0.005)}
              # 'Catboost': CatBoostRegressor(iterations=10, learning_rate=0.01,
              #                               depth=3, l2_leaf_reg=15, bagging_temperature=8,
              #                               loss_function='MAPE',
              #                               eval_metric='MAPE')

# full_data = pd.read_csv('spanish_league.csv')
class models():

    def __init__(self, league_name):
        self.sub_league = full_data[full_data.League_x == league_name]
        self.drop_col = ['League_x', 'Home', 'Away', 'Time',
                         'Start_Sales', 'Close_Sales', 'Date']
        self.one_hot_cols = ['Home', 'Away']
        self.league_name = league_name

    def is_live_transf(self):
        self.sub_league['Is_live'] = (self.sub_league['Is_Live'] == 'Yes').astype(int)

    def one_hot_encode(self):
        one_hot_data = pd.concat([pd.get_dummies(self.sub_league[col]) for col in self.one_hot_cols], axis=1)
        self.sub_league = pd.concat((self.sub_league, one_hot_data), axis=1)
        print 'Dimension after one-hot-encoding is', self.sub_league.shape

    def drop_cols(self):
        self.sub_league.drop(self.drop_col, axis=1, inplace=True)
        print 'Dimension after dropping columns is', self.sub_league.shape

    def run_models(self, classifiers):
        print 'League name is', self.league_name
        # self.is_live_transf()
        self.one_hot_encode()
        self.drop_cols()

        # split date
        def mape(y_pred, y_true):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        X_train, X_test, y_train, y_test = train_test_split(self.sub_league.drop('Stake', axis=1),
                                                            self.sub_league['Stake'], test_size=0.33)

        for name, clf in classifiers.items():
            print 'Running', name, '...'
            clf = clf.fit(X_train, np.log(y_train))
            y_pred = clf.predict(X_test)
            acc = 100 - mape(np.exp(y_pred), y_test)
            print('Accuracy is {:.3f}%'.format(acc))


model = models('Spanish League')
model.run_models(classifiers=CLASSFIERS)  # 68%~70%

model = models('Italian League')
model.run_models(classifiers=CLASSFIERS)  # 70%

# take into account the bet type
# match_num_betype = pd.read_csv('match_num_betype.csv')
# merge sp_league with match_betypes by home, away and time
# date = pd.to_datetime(sp_league['Date'])
# sp_league.is_copy = False
# sp_league['date'] = date.dt.date
# sp_league.set_index(sp_league['date'], inplace=True)
# match_num_betype.set_index(match_num_betype['match_time'], inplace=True)

# sp_league.join(match_num_betype, how='inner')

