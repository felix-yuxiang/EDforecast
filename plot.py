import matplotlib.pyplot as plt

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