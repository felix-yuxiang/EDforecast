import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# import shap # SHAP package does not work on python 3.12!
from sklearn.compose import ColumnTransformer

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from prophet import Prophet



df = pd.read_csv('./data/output_data.csv', index_col=0)
df = df[~((df['Date'] >= '2020-03-15') & (df['Date'] < '2020-05-14'))]

X = df['Date']

y = df['Number_Visits'].map(lambda x: int(x.replace(',', '')))

Xy = pd.DataFrame(pd.concat([X, y], axis=1))
Xy.rename(columns={'Date': 'ds', 'Number_Visits' : 'y'}, inplace=True)
#print(Xy)

model = Prophet()
model.fit(Xy)

future = model.make_future_dataframe(periods=365) 

forecast = model.predict(future)

print(forecast)

# fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)

# plt.title('Person Count Prediction with Prophet')
# plt.xlabel('Date')
# plt.ylabel('Person Count')

plt.savefig('./plots/ts_comp.png')