import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
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
fd_result = "./results/deterministic.txt"
with open(fd_result, "a") as f:
        f.write(f"---------------------------Lasso Regression--------------------------\n")

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

### save the result to the following path
# fd_result = "./results/random.txt" if random_split else "./results/deterministic.txt"

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

def plot_with_isweekend(X_test, y_test, y_pred, model_name):
    size=10

    weekend = X_test['is_weekend'] == 1
    weekday = X_test['is_weekend'] == 0

    plt.scatter(X_test.index[weekday], y_pred[weekday], label='Weekdays (y_pred)', alpha=0.5, s=size, color='blue')
    plt.scatter(X_test.index[weekend], y_pred[weekend], label='Weekends (y_pred)', alpha=0.5, s=size, color='red')
    plt.scatter(X_test.index[weekday], y_test[weekday], label='Actual values weekdays(y_test)', alpha=0.5,s=size, color='green')
    plt.scatter(X_test.index[weekend], y_test[weekend], label='Actual values weekend(y_test)', alpha=0.5, s=size, color='orange')

    plt.title(model_name+': Plot of y_test vs. y_pred')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend(loc='upper center', bbox_to_anchor=(0.9, -0.05))
    plt.tight_layout()
    # Set y-axis range
    plt.ylim(2500, 6500)

    # Set y-axis grid lines at intervals of 500
    plt.yticks(range(2500, 6501, 500))
    
    plt.savefig('./plots/'+model_name+'_with_isweekend.png')
    plt.clf()



ct = ColumnTransformer([
        ('weathers scaler', StandardScaler(), ['MIN_TEMPERATURE', 'MEAN_TEMPERATURE', 'MAX_TEMPERATURE', 'TOTAL_SNOW',
       'TOTAL_RAIN', 'TOTAL_PRECIPITATION', 'HEATING_DEGREE_DAYS', 'COOLING_DEGREE_DAYS'])
    ], remainder='passthrough')


rft = TransformedTargetRegressor(regressor = Lasso(alpha=0.1),transformer = StandardScaler())
steps = [("scaler", ct),("regressor", rft)]
piplasso = Pipeline(steps)

ct = ct.fit(X_train, y_train)
X_transformed = ct.transform(X_train)
X_train_transformed = pd.DataFrame(X_transformed, columns=X_train.columns)
mat = ct.transform(X_test)
X_test_transformed = pd.DataFrame(mat, columns=X_test.columns)
# print(X_transformed)


# Parameter tuning
param_grid = {
    'regressor__regressor__alpha': [0.01, 0.1, 1.0, 10.0]
}


# grid_search = GridSearchCV(estimator=piplasso, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
# grid_search.fit(X_train, y_train)

# print("Best Hyperparameters:", grid_search.best_params_)

# best_model = grid_search.best_estimator_


piplasso.fit(X_train, y_train)
y_pred = piplasso.predict(X_test)
y_pred_train = piplasso.predict(X_train_bc)
mse_train = mean_squared_error(y_train_bc, y_pred_train)
mad_train = mean_absolute_error(y_train_bc, y_pred_train)
mse = mean_squared_error(y_test, y_pred)
mad = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Training Error: {mse_train:.2f}')
print(f'Mean Absolute Training Error: {mad_train:.2f}')
print(f'Mean Squared Testing Error: {mse:.2f}')
print(f'Mean Absolute Testing Error: {mad:.2f}')

# with open(fd_result, "a") as f:
#     f.write(str(grid_search.best_params_))

with open(fd_result, "a") as f:
    f.write(f"\n Lasso with Tuning Hyperparameter: \n Mean Squared Training Error: {mse_train:.2f} \n Mean Absolute Training Error: {mad_train:.2f} \n Mean Squared Error: {mse:.2f} \n Mean Absolute Error: {mad:.2f} \n")

# weekend = X_test['is_weekend'] == 1
# weekday = X_test['is_weekend'] == 0

# plt.scatter(X_test.index[weekday], y_pred[weekday], label='Weekdays (y_pred)', alpha=0.5, color='blue')
# plt.scatter(X_test.index[weekend], y_pred[weekend], label='Weekends (y_pred)', alpha=0.5, color='red')
# plt.scatter(X_test.index[weekday], y_test[weekday], label='Actual values weekdays(y_test)', alpha=0.5, color='green')
# plt.scatter(X_test.index[weekend], y_test[weekend], label='Actual values weekend(y_test)', alpha=0.5, color='orange')

# plt.scatter(X_test.index, y_test, label='Actual values (y_test)')
# plt.scatter(X_test.index, y_pred, label='Predicted values (y_pred)')
# plt.title('Lasso: Plot of y_test vs. y_pred on is_weekend')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend()
    
# plt.savefig('./plots/lasso_isweekend.png')
# plot_with_isweekend(X_test, y_test, y_pred, "Lasso")

# Lasso feature selection
# lasso_coef = np.abs(best_model.named_steps['regressor'].regressor_.coef_)

# names=X_train.columns
# feature_subset=np.array(names)[lasso_coef>0.001]

# feature_subset = pd.DataFrame({'feature':np.array(names)[lasso_coef>0.001], 'coef': lasso_coef[lasso_coef>0.001]})
# print(feature_subset)
# plt.bar(names, lasso_coef)
# plt.xticks(rotation=90)
# plt.grid()
# plt.title("Feature Selection Based on Lasso")
# plt.xlabel("Features")
# plt.ylabel("Importance")
# plt.ylim(0, 0.15)

# plt.savefig('./plots/lasso_feature.png')




