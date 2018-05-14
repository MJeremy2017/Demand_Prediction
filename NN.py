import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


full_data = pd.read_csv('full_data.csv')  # is_live has been converted
full_data.columns


class model_nn():

    def __init__(self, league_name):
        self.sub_league = full_data[full_data.League_x == league_name]
        self.drop_col = ['League_x', 'Home', 'Away', 'Time',
                         'Start_Sales', 'Close_Sales', 'Date']
        self.one_hot_cols = ['Home', 'Away']
        self.league_name = league_name

    def one_hot_encode(self):
        one_hot_data = pd.concat([pd.get_dummies(self.sub_league[col]) for col in self.one_hot_cols], axis=1)
        self.sub_league = pd.concat((self.sub_league, one_hot_data), axis=1)
        print('Dimension after one-hot-encoding is', self.sub_league.shape)

    def drop_cols(self):
        self.sub_league.drop(self.drop_col, axis=1, inplace=True)
        print('Dimension after dropping columns is', self.sub_league.shape)

    def get_model(self):
        MODEL = Sequential()
        MODEL.add(Dense(200, input_dim=self.sub_league.shape[1]-1, activation='relu'))
        MODEL.add(Dense(100, activation='relu'))
        MODEL.add(Dense(32, activation='relu'))
        MODEL.add(Dense(1))
        MODEL.compile(loss='mean_squared_error', optimizer='adam')
        return MODEL

    def run_models(self):
        print('League name is', self.league_name)
        self.one_hot_encode()
        self.drop_cols()
        MODEL = self.get_model()

        # split date
        def mape(y_pred, y_true):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        X_train, X_test, y_train, y_test = train_test_split(self.sub_league.drop('Stake', axis=1),
                                                            self.sub_league['Stake'], test_size=0.33)

        # running models
        MODEL.fit(X_train, np.log(y_train), epochs=5, batch_size=32)
        print 'Training finished'
        y_pred = MODEL.predict(X_test, batch_size=32)
        acc = 100 - mape(np.exp(y_pred), y_test)
        print 'Accuracy is {0}%'.format(acc)


model = model_nn('Spanish League')
model.run_models()  # bad result

# ============ redo ============= #
train = model.sub_league
print train.columns


class model_nn2():
    def __init__(self, train):
        self.num_col = train.shape[1]

    def get_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(train.drop('Stake', axis=1),
                                                            train['Stake'], test_size=0.33)
        return X_train, X_test, y_train, y_test

    def normalize(self, dat):
        dat_norm = dat.apply(lambda s: s/np.max(s))
        return dat_norm

    def get_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.num_col-1))
        model.add(Activation('sigmoid'))
        model.add(Dropout(0.15))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.15))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def run_model(self):
        def mape(y_pred, y_true):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        X_train, X_test, y_train, y_test = self.get_train_test()
        X_train = self.normalize(X_train)
        X_test = self.normalize(X_test)

        MODEL = self.get_model()
        MODEL.fit(X_train, np.log(y_train), epochs=5, batch_size=32)
        print 'Training finished'
        y_pred = MODEL.predict(X_test, batch_size=32)
        acc = 100 - mape(np.exp(y_pred), y_test)
        print 'Accuracy is {0}%'.format(acc)


model = model_nn2(train)
model.run_model()  # bad result below 40%



