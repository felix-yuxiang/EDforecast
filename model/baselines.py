import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import shap # SHAP package does not work on python 3.12!


df = pd.read_csv('../data/output_data.csv')

X = df.drop(columns=['Date','Number_Visits', 'holiday_name'])
y = df['Number_Visits'].map(lambda x: int(x.replace(',', '')))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

models = {'Random Forest': RandomForestRegressor(n_estimators=1000, random_state=42), 
          'Linear Regression': LinearRegression()}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model:{model_name} Mean Squared Error: {mse:.2f}")


# SHAP
shap.initjs()
rf_model = models['Random Forest']
rf_explainer = shap.Explainer(rf_model)
shap_values = rf_explainer.shap_values(X_test)
# print(len(shap_values[0]))
# shap.summary_plot(shap_values[:,1:], X_test.iloc[:,1:], show=False)
# plt.savefig('../plots/shap_rf_drop_unknown0.png')

# Force plot
shap.plots.force(rf_explainer.expected_value[0], shap_values[0,:], X_test.iloc[0, :], matplotlib = True, show=False)
plt.savefig('../plots/force_plot_rf.png')