import numpy as np
import os
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
from sklearn.model_selection import GridSearchCV



df = pd.read_csv('./data/output_data.csv', index_col=0)
### save the result to the following path
fd_result = "./results/deterministic.txt"

# df = df[~((df['Date'] >= '2020-03-15') & (df['Date'] < '2020-05-14'))]
# with open(fd_result, "a") as f:
#         f.write(f"---------------------------Dropping Outlier Rows--------------------------\n")

# encoding the province
df = pd.get_dummies(df, columns=['Province'], dtype=float)

X = df.drop(columns=['Date','Number_Visits', 'holiday_name', 'normal day'])
y = df['Number_Visits'].map(lambda x: int(x.replace(',', '')))
os.makedirs('./results', exist_ok=True)
# print(X.columns)

# X_cts = X.values

### standardize data + random split 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, y_train = shuffle(X_train, y_train, random_state=42)
# random_split = True

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




# Process Transformer
ct = ColumnTransformer([
        ('weathers_scaler', StandardScaler(), ['MIN_TEMPERATURE', 'MEAN_TEMPERATURE', 'MAX_TEMPERATURE', 'TOTAL_SNOW',
       'TOTAL_RAIN', 'TOTAL_PRECIPITATION', 'HEATING_DEGREE_DAYS', 'COOLING_DEGREE_DAYS'])
    ], remainder='passthrough')


lrt = TransformedTargetRegressor(regressor = LinearRegression(),transformer = StandardScaler())
steps = [("scaler", ct),("regressor", lrt)]
piplinear = Pipeline(steps)


rft = TransformedTargetRegressor(regressor = RandomForestRegressor(n_estimators=1000, random_state=42),transformer = StandardScaler())
steps = [("scaler", ct),("regressor", rft)]
piprand = Pipeline(steps)
pipelines_dct = {'Random Forest with standardization': piprand,
                'Linear Regression with standardization': piplinear}

models = {'Random Forest without standardization': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), 
          'Linear Regression without standardization': LinearRegression()}

ct = ct.fit(X_train, y_train)
X_transformed = ct.transform(X_train)


mat = ct.transform(X_test)
X_test_transformed = pd.DataFrame(mat, columns=X_test.columns)


# Function for plots
def plot_with_isweekend(X_test, y_test, y_pred, model_name):

    weekend = X_test['is_weekend'] == 1
    weekday = X_test['is_weekend'] == 0


    plt.scatter(X_test.index[weekday], y_pred[weekday], label='Weekdays (y_pred)', alpha=0.5, color='blue')
    plt.scatter(X_test.index[weekend], y_pred[weekend], label='Weekends (y_pred)', alpha=0.5, color='red')
    plt.scatter(X_test.index[weekday], y_test[weekday], label='Actual values weekdays(y_test)', alpha=0.5, color='green')
    plt.scatter(X_test.index[weekend], y_test[weekend], label='Actual values weekend(y_test)', alpha=0.5, color='orange')

    # plt.scatter(X_test.index, y_test, label='Actual values (y_test)')
    # plt.scatter(X_test.index, y_pred, label='Predicted values (y_pred)')
    plt.title(model_name+': Plot of y_test vs. y_pred')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    
    plt.savefig('./plots/'+model_name+'_with_isweekend.png')
    plt.clf()


def plot_test_pred(X_test, y_test, y_pred, model_name):
    plt.scatter(X_test.index, y_test, label='Actual values (y_test)')
    plt.scatter(X_test.index, y_pred, label='Predicted values (y_pred)')
    plt.title(model_name+': Plot of y_test vs. y_pred')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
        
    plt.savefig('./plots/'+model_name+'.png')
    plt.clf()


### running the models without standardization
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train_bc)
    mse_train = mean_squared_error(y_train_bc, y_pred_train)
    mad_train = mean_absolute_error(y_train_bc, y_pred_train)
    mse = mean_squared_error(y_test, y_pred)
    mad = mean_absolute_error(y_test, y_pred)

    # with open(fd_result, "a") as f:
    #    f.write(f"Model:{model_name} \n Mean Squared Training Error: {mse_train:.2f} \n Mean Absolute Training Error: {mad_train:.2f} \n Mean Squared Error: {mse:.2f} \n Mean Absolute Error: {mad:.2f} \n\n")

        
    # # Plot y_test and y_pred
    # plot_with_isweekend(X_test,y_test,y_pred,model_name)
    # plot_test_pred(X_test,y_test,y_pred,model_name)


for model_name, model in pipelines_dct.items():
    model.fit(X_train, y_train)
    regressor = model.named_steps['regressor'].regressor_
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train_bc)
    mse_train = mean_squared_error(y_train_bc, y_pred_train)
    mad_train = mean_absolute_error(y_train_bc, y_pred_train)
    mse = mean_squared_error(y_test, y_pred)
    mad = mean_absolute_error(y_test, y_pred)

    # with open(fd_result, "a") as f:
    #     f.write(f"Pipelines:{model_name} \n Mean Squared Training Error: {mse_train:.2f} \n Mean Absolute Training Error: {mad_train:.2f} \n Mean Squared Error: {mse:.2f} \n Mean Absolute Error: {mad:.2f} \n")
    #     # f.write(X_transformed)
    
    # # Plot the y_pred and y_test
    # plot_with_isweekend(X_test,y_test,y_pred,model_name)
    # plot_test_pred(X_test,y_test,y_pred,model_name)

    feature_importance = np.array([])
    if model_name=='Linear Regression with standardization':
        feature_importance = model.named_steps['regressor'].regressor_.coef_
    else:
        feature_importance = model.named_steps['regressor'].regressor_.feature_importances_
    
    # feature_importance = np.sort(feature_importance)[::-1]
    feature_name = X_test.columns
    feature_df = pd.DataFrame({'Feature':feature_name, 'Importance':feature_importance}).sort_values(by=['Importance'],ascending=False)
    
    # with open(fd_result, "a") as f:
    #     f.write(f"Pipelines:{model_name}\n Feature Important: {feature_df}\n\n")
    # print(len(feature_importance))
    #SHAP

    # shap_explainer = shap.Explainer(regressor)
    # shap_values = shap_explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test, show=False)
    # plt.savefig('./plots/shap_'+model_name+'.png')


# # Hyperparameter tuning

param_grid_rf = {
    'regressor__regressor__n_estimators': [5,10,15],
    'regressor__regressor__max_depth': [None, 10, 20],
    'regressor__regressor__min_samples_split': [2, 5, 10],
    'regressor__regressor__min_samples_leaf': [1, 2, 4],
}

# with open(fd_result, "a") as f:
#         f.write(f"---------------------------Random Forest Tuning Hyperparameter--------------------------\n")

grid_search_rf = GridSearchCV(estimator=piprand, param_grid=param_grid_rf, scoring='neg_mean_absolute_error', cv=3)
grid_search_rf.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search_rf.best_params_)
best_model_rf = grid_search_rf.best_estimator_

best_model_rf.fit(X_train, y_train)
y_pred = best_model_rf.predict(X_test)
y_pred_train = best_model_rf.predict(X_train_bc)
mse_train_best = mean_squared_error(y_train_bc, y_pred_train)
mad_train_best = mean_absolute_error(y_train_bc, y_pred_train)
mse_best = mean_squared_error(y_test, y_pred)
mad_best = mean_absolute_error(y_test, y_pred)

plot_test_pred(X_test, y_test,y_pred, "Random Forest with Tuned Hyperparameter")


# with open(fd_result, "a") as f:
#     f.write(f"Pipelines:Random Forest with standardization and hyperparameter tuning \n Mean Squared Training Error: {mse_train_best:.2f} \n Mean Absolute Training Error: {mad_train_best:.2f} \n Mean Squared Error: {mse:.2f} \n Mean Absolute Error: {mad:.2f} \n")

# province_name = ['Province_BC', 'Province_ON', 'Province_QC']
# col_index = [0,0,0]
# for i in range(len(province_name)):
#     col_index[i] = col_index.append(X_test.columns.get_loc(province_name[i]))
# col_index = col_index[3:6]
# print(col_index)


# SHAP on Random Forest
# shap.initjs()
# rf_explainer = shap.Explainer(piprand.named_steps['regressor'].regressor_)
# shap_values = rf_explainer.shap_values(X_test)
# shap_unselected = np.delete(shap_values, col_index, axis=1)
# X_test_unselected = X_test.drop(columns=province_name)
# print(len(shap_unselected[0]))
# # shap.summary_plot(shap_values, X_test, show=False)
# # plt.savefig('./plots/shap_rf.png')
# # plt.clf()

# # remove province
# # print(len(shap_values[0]))
# # print(len(X_test))
# shap.summary_plot(shap_unselected, X_test_unselected, show=False)
# plt.savefig('./plots/shap_rf_remove_province.png')
# # plt.clf()

# # # Force plot
# # shap.plots.force(rf_explainer.expected_value[0], shap_values[0,:], X_test.iloc[0, :], matplotlib = True, show=False)
# # plt.savefig('../plots/force_plot_rf.png')