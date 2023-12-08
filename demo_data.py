import pandas as pd

data = pd.read_csv('data/output_data_demographic.csv')
data_bc = data[data['Province']=='BC']
print(data.info())
print(data_bc.shape)
print(data[data['Date']=='2019-03-01'])
data_20190301 = data[data['Date']=='2019-03-01']
test = data.loc[1600:]
test_bc = test[test['Province']=='BC'].drop_duplicates()
print(test_bc[['Date']].head(150))
