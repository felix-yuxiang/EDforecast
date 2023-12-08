import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  

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
    
    plt.savefig('./plots/vanilla_'+model_name+'.png')
    plt.clf()


df = pd.read_csv('./data/output_data.csv', index_col=0)
# df = df[~((df['Date'] >= '2020-03s-15') & (df['Date'] < '2020-05-14'))]
y_BC = df[df['Province']=='BC']['Number_Visits'].map(lambda x: int(x.replace(',', '')))
y_ON = df[df['Province']=='ON']['Number_Visits'].map(lambda x: int(x.replace(',', '')))
y_QC = df[df['Province']=='QC']['Number_Visits'].map(lambda x: int(x.replace(',', '')))
y = df['Number_Visits'].map(lambda x: int(x.replace(',', '')))  
# plt.plot(y_BC)
# plt.plot(y_ON)
# plt.plot(y_QC)
# plt.legend(['BC','ON','QC'])
# plt.ylabel('Number of visits')
# plt.xlabel('Chronological Index')
# plt.savefig('./plots/y.png')

plt.hist(y_BC, bins=30, alpha=0.7, label='BC')
plt.hist(y_ON, bins=30, alpha=0.7, label='ON')
plt.hist(y_QC, bins=30, alpha=0.7, label='QC')

# y.plot.kde(label='KDE', color = 'red')
# plt.hist(posterior_samples, bins=30, density=True, alpha=0.5, color='blue', label='Posterior Samples')
# plt.ticklabel_format(style='sci', axis='x', scilimits=(3,4))
# plt.ticklabel_format(style='sci', axis='y', scilimits=(-4,-3))
plt.title('Data distribution')
plt.xlabel('Number of ED visits')
plt.ylabel('Frequency')
plt.legend(title = 'Province')
plt.savefig('./results/data_hist_pop.png')