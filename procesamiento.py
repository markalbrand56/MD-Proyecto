import pandas as pd

data  = pd.read_csv('merged_data.csv', sep=',', encoding='utf-8', index_col=0)
train = data[data['Year'] >= 2017]
data = data[data['Year'] < 2017]

#exportar a csv
data.to_csv('data_procesada.csv', sep=',', encoding='utf-8')
train.to_csv('train_procesada.csv', sep=',', encoding='utf-8')
