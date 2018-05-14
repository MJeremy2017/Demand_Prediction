import pandas as pd
import numpy as np

dat = pd.read_csv('C:\Users\ZhangYue\Desktop\data\dat.csv')
dat.head()

dat['League'].nunique()

dat = dat[~dat['Bet type'].str.contains('HTH')]

d = {
    'year': [2000, 2000, 2000, 2000, 2001, 2001, 2001],
    'team': ['A', 'B', 'B', 'A', 'B', 'A', 'A'],
    'value': [1, 0, 0, 1, 2, 3, 3],
}

df = pd.DataFrame(d)

df['mean_per_team_and_year'] = df.groupby(['team', 'year']).transform('mean')
print(df)

df = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'],
                  index=pd.date_range('1/1/2000', periods=10))
df.iloc[3:7] = np.nan

df.transform(lambda x: (x - x.mean()) / x.std())

