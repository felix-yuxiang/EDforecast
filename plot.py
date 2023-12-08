import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
import numpy as np

df = pd.read_csv('data/output_data.csv')
df = pd.get_dummies(df, columns=['Province'])

df = df.sort_values(by='Date')

X = df.drop(columns=['Date','Number_Visits', 'holiday_name', 'normal day'])
y = df['Number_Visits'].map(lambda x: int(x.replace(',', '')))

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
train_index_on = X_train['Province_ON'] == 1
X_train_on = X_train[train_index_on]
train_index_qc = X_train['Province_QC'] == 1
X_train_qc = X_train[train_index_qc]

# X_train, y_train = shuffle(X_train, y_train, random_state=42)
random_split = False
print(df[['TOTAL_SNOW','TOTAL_RAIN']].describe())

numerical_features = ['MIN_TEMPERATURE','MEAN_TEMPERATURE','MAX_TEMPERATURE','TOTAL_SNOW','TOTAL_RAIN','TOTAL_PRECIPITATION',
    'HEATING_DEGREE_DAYS','COOLING_DEGREE_DAYS']

# Spline_bc = make_interp_spline(X_train_bc.index, X_train_bc['MEAN_TEMPERATURE'])
# Spline_on = make_interp_spline(X_train_on.index, X_train_on['MEAN_TEMPERATURE'])
# Spline_qc = make_interp_spline(X_train_qc.index, X_train_qc['MEAN_TEMPERATURE'])
 
# # Returns evenly spaced numbers
# # over a specified interval.
# X_bc = np.linspace(X_train_bc.index.min(), X_train_bc.index.max(), 100)
# X_on = np.linspace(X_train_on.index.min(), X_train_on.index.max(), 100)
# X_qc = np.linspace(X_train_qc.index.min(), X_train_qc.index.max(), 100)
# Y_bc = Spline_bc(X_bc)
# Y_on = Spline_bc(X_on)
# Y_qc = Spline_bc(X_qc)

# plt.plot(X_train_on.index, X_train_on['MEAN_TEMPERATURE'], linestyle='-',linewidth=1,alpha=0.6,color='blue',label='ON')
# plt.plot(X_train_qc.index, X_train_qc['MEAN_TEMPERATURE'], linestyle='-',linewidth=1,alpha=0.6,color='orange',label='QC')
# plt.plot(X_train_bc.index, X_train_bc['MEAN_TEMPERATURE'], linestyle='-',linewidth=1,alpha=0.6,color='green',label='BC')
# # plt.plot(X_bc, Y_bc, linestyle='-',linewidth=1,alpha=0.6,color='green',label='BC')
# # plt.plot(X_on, Y_on, linestyle='-',linewidth=1,alpha=0.6,color='blue',label='ON')
# # plt.plot(X_qc, Y_qc, linestyle='-',linewidth=1,alpha=0.6,color='orange',label='QC')
# plt.xlabel('Chronological Index')
# plt.ylabel('Mean Temperature')
# plt.title('Line Graph of Temperature of Each Province')
# plt.legend(loc='best')
# # plt.savefig('./plots/temperature.png')
# # plt.show()
# plt.clf()
    
plt.hist(X_train['TOTAL_SNOW'], bins=50,alpha=0.6,color='blue',label='Total Snow')
plt.hist(X_train['TOTAL_RAIN'], bins=50,alpha=0.5,color='orange',label='Total Rain')
# plt.hist(X_train_on['TOTAL_RAIN'], bins=50,alpha=0.5,color='orange',label='Total Rain in ON')
# plt.hist(X_train_qc['TOTAL_RAIN'], bins=50,alpha=0.5,color='blue',label='Total Rain in QC')
plt.hist(X_train['TOTAL_PRECIPITATION'], bins=50,alpha=0.4,color='green',label='Total Precipitation')
plt.xlabel('Precipitation')
plt.ylabel('Density')
plt.title('Histogram of Total Rain, Snow, and Precipitation')
plt.xlim(-1, 25)
plt.legend(loc='best')
plt.savefig('./plots/snow_rain.png')
# plt.show()
plt.clf()

# plt.plot(X_train.index, X_train['HEATING_DEGREE_DAYS'], linestyle='-',linewidth=0.5,alpha=0.6,color='blue',label='Heating Degree Days')
# plt.plot(X_train.index, X_train['COOLING_DEGREE_DAYS'], linestyle='-',linewidth=0.5,alpha=0.6,color='orange',label='Cooline Degree Days')
# plt.xlabel('Chronological Index')
# plt.ylabel('Degree Days')
# plt.title('Line Graph of Degree Days')
# plt.legend(loc='best')
# plt.show()
# plt.savefig('./plots/temperature.png')