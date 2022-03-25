import pandas as pd
x = pd.read_csv('data/input_train.csv')
x['sample_id'] = x['cluster'] * 21 + x['day']
print(len(x.groupby('sample_id')))
