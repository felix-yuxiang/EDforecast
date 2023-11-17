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



# df = pd.read_csv('./data/output_data_demographic.csv', index_col=0)
df = pd.read_csv('./data/output_data.csv', index_col=0)
df = df[~((df['Date'] >= '2020-03-15') & (df['Date'] < '2020-05-14'))]
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
X_train, y_train = shuffle(X_train, y_train, random_state=42)
random_split = False

### save the result to the following path
fd_result = "./results/random.txt" if random_split else "./results/deterministic.txt"

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

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

models = {'Random Forest': RandomForestRegressor(n_estimators=1000, random_state=42), 
          'Linear Regression': LinearRegression()}

# print('Training data before the column transformer:')
# print(X_train.head())   
ct = ct.fit(X_train, y_train)
X_transformed = ct.transform(X_train)
# print('Transformed data:')
# print(X_transformed)
# print(X_transformed.shape)




### running the models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mad = mean_absolute_error(y_test, y_pred)
    with open(fd_result, "a") as f:
       f.write(f"Model:{model_name} \n Mean Squared Error: {mse:.2f} \n Mean Absolute Error: {mad:.2f} \n")

# shap.initjs()
for model_name, model in pipelines_dct.items():
    model.fit(X_train, y_train)
    regressor = model.named_steps['regressor'].regressor_
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mad = mean_absolute_error(y_test, y_pred)
    with open(fd_result, "a") as f:
        f.write(f"Pipelines:{model_name} with standardized features. \n Mean Squared Error: {mse:.2f} \n Mean Absolute Error: {mad:.2f} \n")
        # f.write(X_transformed)
    plt.scatter(y_test, y_pred)
    feature_importance = np.array([])
    if model_name=='Linear Regression':
        feature_importance = model.named_steps['regressor'].regressor_.coef_
    else:
        feature_importance = model.named_steps['regressor'].regressor_.feature_importances_
    
    # feature_importance = np.sort(feature_importance)[::-1]
    feature_name = X_test.columns
    feature_df = pd.DataFrame({'Feature':feature_name, 'Importance':feature_importance}).sort_values(by=['Importance'],ascending=False)
    with open(fd_result, "a") as f:
        f.write(f"Pipelines:{model_name} with standardized features. \n Feature Important: {feature_df}\n")
    # print(len(feature_importance))
    #SHAP

    # shap_explainer = shap.Explainer(regressor)
    # shap_values = shap_explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test, show=False)
    # plt.savefig('./plots/shap_'+model_name+'.png')

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