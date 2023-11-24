import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# import shap # SHAP package does not work on python 3.12!
from sklearn.compose import ColumnTransformer

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# import xgboost as xgb

df = pd.read_csv('./data/output_data.csv', index_col=0)
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
X_train, y_train = shuffle(X_train, y_train, random_state=42)
random_split = False

### save the result to the following path
# fd_result = "./results/random.txt" if random_split else "./results/deterministic.txt"

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


ct = ColumnTransformer([
        ('weathers scaler', StandardScaler(), ['MIN_TEMPERATURE', 'MEAN_TEMPERATURE', 'MAX_TEMPERATURE', 'TOTAL_SNOW',
       'TOTAL_RAIN', 'TOTAL_PRECIPITATION', 'HEATING_DEGREE_DAYS', 'COOLING_DEGREE_DAYS'])
    ], remainder='passthrough')


rft = TransformedTargetRegressor(regressor = GradientBoostingRegressor(random_state=42),transformer = StandardScaler())
steps = [("scaler", ct),("regressor", rft)]
pipxg = Pipeline(steps)

ct = ct.fit(X_train, y_train)
X_transformed = ct.transform(X_train)
X_train_transformed = pd.DataFrame(X_transformed, columns=X_train.columns)
mat = ct.transform(X_test)
X_test_transformed = pd.DataFrame(mat, columns=X_test.columns)
# print(X_transformed)


# Parameter tuning
# param_grid = {
#     'regressor__regressor__n_estimators': [50, 100, 200],
#     'regressor__regressor__learning_rate': [0.01, 0.1, 0.2],
#     'regressor__regressor__max_depth': [3, 4, 5],
# }

# grid_search = GridSearchCV(estimator=pipxg, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3)
# grid_search.fit(X_train, y_train)

# print("Best Hyperparameters:", grid_search.best_params_)

# best_model = grid_search.best_estimator_


pipxg.fit(X_train, y_train)
y_pred = pipxg.predict(X_test)
y_pred_train = pipxg.predict(X_train_transformed)
mse_train = mean_squared_error(y_train, y_pred_train)
mad_train = mean_absolute_error(y_train, y_pred_train)
mse = mean_squared_error(y_test, y_pred)
mad = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Training Error: {mse_train:.2f}')
print(f'Mean Absolute Training Error: {mad_train:.2f}')
print(f'Mean Squared Testing Error: {mse:.2f}')
print(f'Mean Absolute Testing Error: {mad:.2f}')



