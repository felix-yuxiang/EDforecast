import pandas as pd

data = pd.read_csv('data/covid.csv')

# Only keep BC data
data_bc = data[data['B.C.\n\nNumber of ED visits, pre-pandemic','B.C.\n\nNumber of ED visits, pandemic period']]