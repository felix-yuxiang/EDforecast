import numpy as np
from numpy import datetime64
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

additional_regressor = True

df = pd.read_csv('./data/output_data.csv', index_col=0)
df = df[~((df['Date'] >= '2020-03-15') & (df['Date'] < '2020-05-14')) & (df['Province'] == "BC")]
df['Date'] = df['Date'].astype('datetime64[ns]')
df.reset_index()

# X = pd.DataFrame(pd.concat([df['Date'], df['is_holiday'], df['MEAN_TEMPERATURE'], df['TOTAL_PRECIPITATION']], axis=1))
print(df.columns)
X = pd.DataFrame(pd.concat( [df['Date'], df['is_holiday'], df['MIN_TEMPERATURE'], df['MAX_TEMPERATURE'], df['TOTAL_SNOW'], df['TOTAL_RAIN'], df['MEAN_TEMPERATURE'], df['TOTAL_PRECIPITATION'], df['is_weekend']], axis=1))
X.rename(columns={'Date': 'ds'}, inplace=True)
X.reset_index()
y = df['Number_Visits'].map(lambda x: int(x.replace(',', '')))
y.reset_index()
y.rename('y', inplace=True)

split_index = int(0.8 * len(X))
predict_period = len(X) - split_index
print(split_index, predict_period)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

Xy = pd.DataFrame(pd.concat([X_train, y_train], axis=1))
# Xy.rename(columns={'Date': 'ds', 'Number_Visits' : 'y'}, inplace=True)
# print(Xy)

model = Prophet()
model.add_country_holidays(country_name='CA')
if additional_regressor: 
    model.add_regressor('is_holiday')
    model.add_regressor('is_weekend')
    model.add_regressor('MIN_TEMPERATURE')
    model.add_regressor('MAX_TEMPERATURE')
    model.add_regressor('TOTAL_SNOW')
    model.add_regressor('TOTAL_RAIN')
    model.add_regressor('MEAN_TEMPERATURE')
    model.add_regressor('TOTAL_PRECIPITATION')
model.fit(Xy)



future = model.make_future_dataframe(periods=predict_period)
# print(X['ds'].info(), future['ds'].info())
if additional_regressor: 
    future = future.merge(X, on='ds',how='left')
# future = pd.DataFrame(pd.concat([future, X_train['MEAN_TEMPERATURE'], X_train['TOTAL_PRECIPITATION']], axis=1))
# print(future)

forecast = model.predict(future)

# print(forecast)
# print(forecast.info())

fig1 = model.plot_components(forecast)
# fig2 = model.plot_components(forecast)

# plt.title('Person Count Prediction with Prophet')
# plt.xlabel('Date')
# plt.ylabel('Person Count')

plt.savefig('./plots/ts/ts_comp_additional_reg.png')
plt.clf()
plt.close()

plt.title('Prediction with Prophet (BC)')
plt.scatter(forecast['ds'], forecast['yhat'], label='Fitted', s=10)
# plt.scatter(y=forecast['yhat'], x=forecast['ds'], color = 'orange')
Date = df['Date'].to_numpy(dtype=datetime64)
plt.scatter(Date,y, color = 'orange', label='Actual',s=10)
plt.legend(loc="lower left")
plt.xticks(rotation=45, ha='right')
print(Date, split_index, Date[split_index])
plt.axvline(x=Date[split_index], color = 'black')
assert(len(forecast['yhat']) == len(y))
mse = mean_squared_error(forecast['yhat'][split_index:], y[split_index:])
mad = mean_absolute_error(forecast['yhat'][split_index:], y[split_index:])
print(mse, mad)
plt.tight_layout()

plt.xlabel('Date')
plt.ylabel('Person Count')

plt.savefig('./plots/ts/prediction_additional_reg.png', bbox_inches="tight")
plt.clf()

# # Define the hyperparameter grid
# p_values = range(0, 2)  # Adjust as needed
# d_values = range(0, 2)  # Adjust as needed
# q_values = range(0, 2)  # Adjust as needed
# P_values = range(0, 2)  # Adjust as needed
# D_values = range(0, 2)  # Adjust as needed
# Q_values = range(0, 2)  # Adjust as needed
# S_values = [7]  # Seasonal period, adjust as needed

# # Create a list of all possible combinations of hyperparameters
# hyperparameters = itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, S_values)

# # Perform grid search
# best_aic = float('inf')
# best_params = None

# for params in hyperparameters:
#     order = params[:3]
#     seasonal_order = params[3:]

#     model = SARIMAX(y_train, exog=X_train[['is_holiday', 'is_weekend', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE', 'TOTAL_SNOW','TOTAL_RAIN', 'MEAN_TEMPERATURE', 'TOTAL_PRECIPITATION']], order=order, seasonal_order=seasonal_order)
#     fit_model = model.fit()

#     forecast_steps = len(y_test)
#     exog_forecast = X_test[['is_holiday', 'is_weekend', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE', 'TOTAL_SNOW','TOTAL_RAIN', 'MEAN_TEMPERATURE', 'TOTAL_PRECIPITATION']]
#     forecast = fit_model.get_forecast(steps=forecast_steps, exog=exog_forecast)

#     rmse = mean_squared_error(y_test, forecast.predicted_mean) 

#     if rmse < best_aic:
#         best_aic = rmse
#         best_params = params


# # Fit the best model
# best_order = best_params[:3]
# best_seasonal_order = best_params[3:]
# print(best_order, best_seasonal_order)


order = (1, 1, 0)  # Example order: SARIMAX(p, d, q)
seasonal_order = (0, 1, 1, 7)  # Example seasonal order: SARIMAX(P, D, Q, S)


# Make predictions
model = SARIMAX(y_train, exog=X_train[['is_holiday', 'is_weekend', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE', 'TOTAL_SNOW','TOTAL_RAIN', 'MEAN_TEMPERATURE', 'TOTAL_PRECIPITATION']], order=order, seasonal_order=seasonal_order)
fit_model = model.fit()
forecast_steps = len(y_test)
exog_forecast = X_test[['is_holiday', 'is_weekend', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE', 'TOTAL_SNOW','TOTAL_RAIN', 'MEAN_TEMPERATURE', 'TOTAL_PRECIPITATION']]
forecast = fit_model.get_forecast(steps=forecast_steps, exog=exog_forecast)

plt.title('Prediction with SARIMAX (BC)')
print(len(Date[split_index:]), len(y_test), len(forecast.predicted_mean))
assert(len(Date[split_index:]) == len(y_test) == len(forecast.predicted_mean))
plt.scatter(Date, pd.concat([pd.Series(y[0]),pd.DataFrame(fit_model.fittedvalues)[1:], forecast.predicted_mean]), label='Fitted', color='orange', s=10)
plt.scatter(Date, y, label = 'Actual', s=10)
plt.axvline(x=Date[split_index], color = 'black')
plt.legend(loc="lower left")
plt.xticks(rotation=45, ha='right')
plt.xlabel('Date')
plt.ylabel('Person Count')
plt.tight_layout()

plt.savefig('./plots/ts/prediction_arima.png', bbox_inches="tight")
plt.clf()

# Evaluate the model
mad = mean_absolute_error(y_test, forecast.predicted_mean)
print(f"Mean Absolute Error (RMSE): {mad}")