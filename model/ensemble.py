import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

df = pd.read_csv('./data/output_data.csv', index_col=0)
fd_result = "./results/deterministic.txt"
with open(fd_result, "a") as f:
        f.write(f"---------------------------Ensemble Model--------------------------\n")

# df = pd.read_csv('./data/output_data.csv', index_col=0)
# df = df[~((df['Date'] >= '2020-03-15') & (df['Date'] < '2020-05-14'))]
# encoding the province
df = pd.get_dummies(df, columns=['Province'])

dates = df['Date']
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


random_forest_regressor = RandomForestRegressor(random_state=42, max_depth=10, min_samples_leaf=2, min_samples_split=5,n_estimators=15)
lasso_regressor = Lasso(random_state=42, alpha=0.01)
ridge_regressor = Ridge(random_state=42, alpha=1.0)

stacking_regressor = StackingRegressor(
    estimators=[
        ('rf', random_forest_regressor),
        ('lasso', lasso_regressor),
        ('ridge', ridge_regressor)
    ],
    final_estimator=TransformedTargetRegressor(
        regressor=RandomForestRegressor(random_state=42),
        transformer=StandardScaler()
    )
)

steps = [("scaler", ct),("regressor", stacking_regressor)]
pipEns = Pipeline(steps)

pipEns.fit(X_train, y_train)
y_pred = pipEns.predict(X_test)
y_pred_train = pipEns.predict(X_train_bc)
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

print(dates[X_test.index])
plt.scatter(dates[X_test.index], y_test, label='Actual values (y_test)',alpha=0.6)
plt.scatter(dates[X_test.index], y_pred, label='Predicted values (y_pred)',alpha=0.6)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gcf().autofmt_xdate(rotation=45)

plt.title('Ensemble: Plot of y_test vs. y_pred')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
    
plt.savefig('./plots/ensemble.png')

with open(fd_result, "a") as f:
    f.write("\n Parameter of RnadomForest: max_depth=10, min_samples_leaf=2, min_samples_split=5,n_estimators=15 \n Parameter of Lasso: alpha=0.01 \n Parameter of Righe: alph=1.0\n")

with open(fd_result, "a") as f:
    f.write(f" Ensemble model: \n Mean Squared Training Error: {mse_train:.2f} \n Mean Absolute Training Error: {mad_train:.2f} \n Mean Squared Error: {mse:.2f} \n Mean Absolute Error: {mad:.2f} \n")

