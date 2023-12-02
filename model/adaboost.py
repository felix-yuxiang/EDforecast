import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('./data/output_data.csv', index_col=0)
fd_result = "./results/deterministic.txt"
with open(fd_result, "a") as f:
        f.write(f"---------------------------Adaboost--------------------------\n")

# df = pd.read_csv('./data/output_data.csv', index_col=0)
# df = df[~((df['Date'] >= '2020-03-15') & (df['Date'] < '2020-05-14'))]
# encoding the province
df = pd.get_dummies(df, columns=['Province'])

X = df.drop(columns=['Date','Number_Visits', 'holiday_name', 'normal day'])
y = df['Number_Visits'].map(lambda x: int(x.replace(',', '')))
os.makedirs('./results', exist_ok=True)

### split the data based on the chronological order, i.e. deterministic split
split_index = int(0.8 * len(X))
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]
index_bc = X_test['Province_BC'] == 1
X_test = X_test[index_bc]
y_test = y_test[index_bc]
train_index_bc = X_train['Province_BC'] == 1
X_train_bc = X_train[train_index_bc]
y_train_bc = y_train[train_index_bc]
X_train, y_train = shuffle(X_train, y_train, random_state=42)
random_split = False

ct = ColumnTransformer([
        ('weathers scaler', StandardScaler(), ['MIN_TEMPERATURE', 'MEAN_TEMPERATURE', 'MAX_TEMPERATURE', 'TOTAL_SNOW',
       'TOTAL_RAIN', 'TOTAL_PRECIPITATION', 'HEATING_DEGREE_DAYS', 'COOLING_DEGREE_DAYS'])
    ], remainder='passthrough')

adaboost = TransformedTargetRegressor(regressor = AdaBoostRegressor(random_state=42, n_estimators=10, loss="exponential"),transformer = StandardScaler())
steps = [("scaler", ct),("regressor", adaboost)]
pipada = Pipeline(steps)

# param_grid = {
#     'regressor__regressor__n_estimators': [5,10,15],
#     'regressor__regressor__learning_rate': [0.01, 0.1, 1.0],
#     'regressor__regressor__loss': ["linear", "square", "exponential"]
# }

# grid_search = GridSearchCV(estimator=pipada, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
# grid_search.fit(X_train, y_train)

# print("Best Hyperparameters:", grid_search.best_params_)

# best_model = grid_search.best_estimator_

pipada.fit(X_train, y_train)
y_pred = pipada.predict(X_test)
y_pred_train = pipada.predict(X_train_bc)
mse_train = mean_squared_error(y_train_bc, y_pred_train)
mad_train = mean_absolute_error(y_train_bc, y_pred_train)
mse = mean_squared_error(y_test, y_pred)
mad = mean_absolute_error(y_test, y_pred)

# weekend = X_test['is_weekend'] == 1
# weekday = X_test['is_weekend'] == 0
# plt.scatter(X_test.index[weekday], y_pred[weekday], label='Weekdays (y_pred)', alpha=0.5, color='blue')
# plt.scatter(X_test.index[weekend], y_pred[weekend], label='Weekends (y_pred)', alpha=0.5, color='red')
# plt.scatter(X_test.index[weekday], y_test[weekday], label='Actual values weekdays(y_test)', alpha=0.5, color='green')
# plt.scatter(X_test.index[weekend], y_test[weekend], label='Actual values weekend(y_test)', alpha=0.5, color='orange')


plt.scatter(X_test.index, y_test, label='Actual values (y_test)',alpha=0.6)
plt.scatter(X_test.index, y_pred, label='Predicted values (y_pred)',alpha=0.6)
plt.title('Adaboost: Plot of y_test vs. y_pred')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
    
plt.savefig('./plots/adaboost.png')

# with open(fd_result, "a") as f:
#     f.write(str(grid_search.best_params_))

with open(fd_result, "a") as f:
    f.write(f"Adaboost: \n Mean Squared Training Error: {mse_train:.2f} \n Mean Absolute Training Error: {mad_train:.2f} \n Mean Squared Error: {mse:.2f} \n Mean Absolute Error: {mad:.2f} \n")

