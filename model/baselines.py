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
import shap # SHAP package does not work on python 3.12!
from sklearn.compose import ColumnTransformer

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV



df = pd.read_csv('./data/output_data.csv', index_col=0)
# df = pd.read_csv('./data/output_data.csv', index_col=0)
# df = df[~((df['Date'] >= '2020-03-15') & (df['Date'] < '2020-05-14'))]
# encoding the province
df = pd.get_dummies(df, columns=['Province'])

X = df.drop(columns=['Date','Number_Visits', 'holiday_name', 'normal day'])
y = df['Number_Visits'].map(lambda x: int(x.replace(',', '')))
os.makedirs('./results', exist_ok=True)
# print(X.columns)

# X_cts = X.values

### standardize data + random split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, y_train = shuffle(X_train, y_train, random_state=42)
random_split = True

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
fd_result = "./results/random.txt" if random_split else "./results/deterministic.txt"

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
# print(X_test['COOLING_DEGREE_DAYS'])
print(X_train.isnull().sum())

ct = ColumnTransformer([
        ('weathers scaler', StandardScaler(), ['MIN_TEMPERATURE', 'MEAN_TEMPERATURE', 'MAX_TEMPERATURE', 'TOTAL_SNOW',
       'TOTAL_RAIN', 'TOTAL_PRECIPITATION', 'HEATING_DEGREE_DAYS', 'COOLING_DEGREE_DAYS'])
    ], remainder='passthrough')


lrt = TransformedTargetRegressor(regressor = LinearRegression(),transformer = StandardScaler())
steps = [("scaler", ct),("regressor", lrt)]
piplinear = Pipeline(steps)


rft = TransformedTargetRegressor(regressor = RandomForestRegressor(n_estimators=1000, random_state=42),transformer = StandardScaler())
steps = [("scaler", ct),("regressor", rft)]
piprand = Pipeline(steps)
pipelines_dct = {'Linear Regression': piplinear,
                'Random Forest': piprand}

models = {'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), 
          'Linear Regression': LinearRegression()}

# print('Training data before the column transformer:')
# print(X_train.head())   
ct = ct.fit(X_train, y_train)
X_transformed = ct.transform(X_train)
# print('Transformed data:')
# print(X_transformed)
# print(X_transformed.shape)

mat = ct.transform(X_test)
X_test_transformed = pd.DataFrame(mat, columns=X_test.columns)

print(pd.DataFrame(X_transformed, columns=X_train.columns))
print(X_test_transformed)


### running the models without standardization
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mad_train = mean_absolute_error(y_train, y_pred_train)
    mse = mean_squared_error(y_test, y_pred)
    mad = mean_absolute_error(y_test, y_pred)
    with open(fd_result, "a") as f:
       f.write(f"Model:{model_name} \n Mean Squared Training Error: {mse_train:.2f} \n Mean Absolute Training Error: {mad_train:.2f} \n Mean Squared Error: {mse:.2f} \n Mean Absolute Error: {mad:.2f} \n\n")

        
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
    
    plt.savefig('./plots/vanilla_'+model_name+'.png')
    plt.clf()


for model_name, model in pipelines_dct.items():
    model.fit(X_train, y_train)
    regressor = model.named_steps['regressor'].regressor_
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mad = mean_absolute_error(y_test, y_pred)
    with open(fd_result, "a") as f:
        f.write(f"Pipelines:{model_name} with standardized features. \n Mean Squared Error: {mse:.2f} \n Mean Absolute Error: {mad:.2f} \n")
        # f.write(X_transformed)
    
    # weekend = X_test['is_weekend'] == 1
    # weekday = X_test['is_weekend'] == 0


    # plt.scatter(X_test.index[weekday], y_pred[weekday], label='Weekdays (y_pred)', alpha=0.5, color='blue')
    # plt.scatter(X_test.index[weekend], y_pred[weekend], label='Weekends (y_pred)', alpha=0.5, color='red')
    # plt.scatter(X_test.index[weekday], y_test[weekday], label='Actual values weekdays(y_test)', alpha=0.5, color='green')
    # plt.scatter(X_test.index[weekend], y_test[weekend], label='Actual values weekend(y_test)', alpha=0.5, color='orange')

    # # plt.scatter(X_test.index, y_test, label='Actual values (y_test)')
    # # plt.scatter(X_test.index, y_pred, label='Predicted values (y_pred)')
    # plt.title(model_name+': Plot of y_test vs. y_pred')
    # plt.xlabel('Index')
    # plt.ylabel('Values')
    # plt.legend()
    
    # plt.savefig('./plots/'+model_name+'.png')
    # plt.clf()

    feature_importance = np.array([])
    if model_name=='Linear Regression':
        feature_importance = model.named_steps['regressor'].regressor_.coef_
    else:
        feature_importance = model.named_steps['regressor'].regressor_.feature_importances_
    
    # feature_importance = np.sort(feature_importance)[::-1]
    feature_name = X_test.columns
    feature_df = pd.DataFrame({'Feature':feature_name, 'Importance':feature_importance}).sort_values(by=['Importance'],ascending=False)
    with open(fd_result, "a") as f:
        f.write(f"Pipelines:{model_name} with standardized features. \n Feature Important: {feature_df}\n\n")
    # print(len(feature_importance))
    #SHAP

    # shap_explainer = shap.Explainer(regressor)
    # shap_values = shap_explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test, show=False)
    # plt.savefig('./plots/shap_'+model_name+'.png')

# Hyperparameter tuning

param_grid_rf = {
    'regressor__regressor__n_estimators': [50, 100, 200],
    'regressor__regressor__max_depth': [None, 10, 20],
    'regressor__regressor__min_samples_split': [2, 5, 10],
    'regressor__regressor__min_samples_leaf': [1, 2, 4],
}




# grid_search_rf = GridSearchCV(estimator=model, param_grid=param_grid_rf, scoring='neg_mean_absolute_error', cv=3)
# grid_search_rf.fit(X_train, y_train)
# print("Best Hyperparameters:", grid_search_rf.best_params_)
# best_model_rf = grid_search_rf.best_estimator_

# best_model_rf.fit(X_train, y_train)
# y_pred = best_model_rf.predict(X_test)
# mse_best = mean_squared_error(y_test, y_pred)
# mad_best = mean_absolute_error(y_test, y_pred)
# with open(fd_result, "a") as f:
#     f.write(f"Pipelines: Random Forest Best Model with standardized features. \n Mean Squared Error: {mse_best:.2f} \n Mean Absolute Error: {mad_best:.2f} \n")




plt.legend(pipelines_dct.keys())
plt.savefig('./plots/piplines_pred_time_split.png')



# SHAP
# shap.initjs()
# rf_model = models['Random Forest']
# rf_explainer = shap.Explainer(rf_model)
# shap_values = rf_explainer.shap_values(X_test)
# # print(len(shap_values[0]))
# # shap.summary_plot(shap_values[:,1:], X_test.iloc[:,1:], show=False)
# # plt.savefig('../plots/shap_rf_drop_unknown0.png')

# # Force plot
# shap.plots.force(rf_explainer.expected_value[0], shap_values[0,:], X_test.iloc[0, :], matplotlib = True, show=False)
# plt.savefig('../plots/force_plot_rf.png')