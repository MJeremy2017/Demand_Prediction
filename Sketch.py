import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import datetime
from workalendar.asia import Singapore
from keras.models import Sequential
from catboost import Pool
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
import numpy as np
import pypyodbc


# Initialize data
cat_features = [0,1,2]
train_data = [["a","b",1,4,5,6],["a","b",4,5,6,7],["c","d",30,40,50,60]]
test_data = [["a","b",2,4,6,8],["a","d",1,4,50,60]]
train_labels = [10,20,30]
# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=2, learning_rate=1, depth=2)
# Fit model
model.fit(train_data, train_labels, cat_features)
# Get predictions
preds = model.predict(test_data)

cat_features = [0,1,2]
train_data = [["a","b",1,4,5,6],["a","b",4,5,6,7],["c","d",30,40,50,60]]
train_labels = [1,1,-1]
test_data = [["a","b",2,4,6,8],["a","d",1,4,50,60]]
# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='Logloss')
# Fit model
model.fit(train_data, train_labels, cat_features)
# Get predicted classes
preds_class = model.predict(test_data)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(test_data)
# Get predicted RawFormulaVal
preds_raw = model.predict(test_data, prediction_type='RawFormulaVal')

train_data = np.random.randint(0, 100, size=(100, 10))
train_label = np.random.randint(0, 1000, size=(100))
test_data = np.random.randint(0, 100, size=(50, 10))
# initialize Pool
train_pool = Pool(train_data, train_label, cat_features=[0,2,5])
test_pool = Pool(test_data, cat_features=[0,2,5])

# specify the training parameters
model = CatBoostRegressor(iterations=2, depth=2, learning_rate=1, loss_function='RMSE')
#train the model
model.fit(train_pool)
# make the prediction using the resulting model
preds = model.predict(test_pool)
print(preds)

# load in the data set with some columns already been deleted
raw_data = pd.read_csv('dataTrimmed.csv')

print(raw_data.Time[1])
type(raw_data.Time[1])
# 'Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p'
datetime.datetime.strptime(raw_data.Time[1], '%m/%d/%Y %I:%M')

d1 = datetime.datetime.strptime('12/1/2010 18:00', '%m/%d/%Y %H:%M')

dt_str = '9/24/2010 5:03'
d2 = datetime.datetime.strptime(dt_str, '%m/%d/%Y %I:%M')

a = raw_data.Time[1].split(' ')
a[0].split('/')
a[1].split(':')
datetime.datetime.today().isoweekday()

sg = Singapore()

sg.is_working_day(datetime.date(2012, 1, 1))
sg.is_working_day(datetime.date(2018, 2, 19))

a, b, c, d, e = split_date('Time')

a = full_data
aa = pd.get_dummies(a)
league = a['Home']
aa = pd.get_dummies(league)

b = pd.concat([pd.get_dummies(a[col]) for col in ['Home', 'Away']], axis=1)

datetime.date(2012, 12, 1)

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
sp_league.Away.nunique()

(sp_league['Is_Live'] == 'Yes').astype(int)

enc = OneHotEncoder()
enc.fit(['yes', 'no'])
enc.transform(['yes'])
enc.fit(sp_league['Home'])
enc.transform(sp_league['Home'])

import re
recom = re.search('\d', '213fdsa')
recom.group()

re.sub('\d', '', '213dfa')

conn = pypyodbc.connect(
    "Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
    "Dbq=D:\R\CustomerPrediction\Odds - 2010 to 2016 Oct.accdb;")

conn = pypyodbc.connect(
    r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};" +
    r"Dbq=C:\Users\Ju\Desktop\Dark Summoner.accdb;")

q = "select * from sprtOdds WHERE `Bet Type` = '1X2'"
data = pd.read_sql(q, conn)

del data
import gc
gc.collect()

def sq():
    i = raw_input('Input an interger: ')
    while int(i)**2 < 50:
        i = raw_input('Input an interger: ')
    print(int(i)**2)

sq()

98356779/746230677.0
516*0.13


# the first person starts with number start_with
def call_out(safii_list, start_with=1):
    n = len(safii_list)
    # get the index of that list
    index = range(n)
    if n <= 1:
        return safii_list
    delete_idx = []
    for i in index:
        if (start_with+i)%3 == 0:
            # pop out the number 3n
            delete_idx.append(i)
    # delete items
    for i in sorted(delete_idx, reverse=True):
        del safii_list[i]
    print(safii_list)
    # get the new start_with
    # last number is start_with + n
    if (start_with+n-1)%3 == 0:
        start_with = 1
    else:
        start_with = (start_with+n-1)%3 + 1

    call_out(safii_list, start_with=start_with)


call_out(range(10))


def number_off(people):
    # go through 1, 2, 3 of the list
    # if reach the end of the list then restart
    starts = 1
    while len(people) > 1:
        people_copy = people
        people_index = range(len(people_copy))
        for i in people_index:
            if (starts+i)%3 == 0:
                current_number = 3
            else:
                current_number = (starts+i)%3
            if current_number == 3:
                people.remove(people_copy[i])
        starts = current_number+1
    return people


number_off(range(10))

# last time there are x peaches, and left y peaches
# (x-1)*(4/5) = y => y*(5/4)+1 = x
# at least how many peaches, which means the last potion is 1 peach
def peach(num_monkey):
    count = 1
    initial_y = 0  # the initial amount peaches, keep testing
    while count < num_monkey:
        # reset
        last_y = initial_y
        for i in range(num_monkey):
            if last_y%4 == 0:
                x = last_y*(5/4.) + 1
                last_y = x
                count += 1
            else:
                # if there is one y doesn't satisfy in the process, set the initial amount+1
                initial_y += 1
                count = 1
                break
    print x

peach(5)

import random

random.choice([0, 1], 4)
random.sample([0, 1], 4)
random.randint(1, 4)


ts = pd.DataFrame({'a': [random.randint(1, 3) for _ in range(10)],
                   'c': [random.choice([0, 1]) for _ in range(10)]})


def cc():
    for row in range(1, ts.shape[0]-1):
        if row > 0 & row < 9:
            cur_row = ts.iloc[row]
            if cur_row['c'] == 1:
                print 'innnnn'
                if ts.iloc[row - 1]['a'] == cur_row['a']:
                    ts.iloc[row - 1]['c'] = 1
                elif ts.iloc[row + 1]['a'] == cur_row['a']:
                    ts.iloc[row + 1]['c'] = 1
    return ts


c




