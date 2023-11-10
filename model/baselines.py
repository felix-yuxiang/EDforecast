import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle

seed = 42

df = pd.read_csv('./data/output_data.csv')

X = df.drop(columns=['Date','Number_Visits', 'holiday_name'])
y = df['Number_Visits'].map(lambda x: int(x.replace(',', '')))

threshold = X.shape[0] * 0.8
print(threshold)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = seed)

X_train = X[:int(threshold)]
X_test = X[int(threshold):] 
y_train = y[:int(threshold)]
y_test = y[int(threshold):] 

plt.plot(y)
plt.savefig('./plots/visits.png')
X_train, y_train = shuffle(X_train, y_train, random_state = seed)

models = {'Random Forest': RandomForestRegressor(n_estimators=1000, random_state = seed), 
          'Linear Regression': LinearRegression()}


for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model:{model_name}")
    print(f"Mean Squared testing Error: {mse:.2f}")
    plt.clf()
    plt.scatter(y_test, y_pred)
    plt.savefig(f'./plots/{model_name}.png')

