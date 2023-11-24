import numpy as np
import pandas as pd
import datetime as dt
import holidays

### Build up the holiday features in BC 
def holiday_feature_onehot(df):
    bc_holidays = holidays.country_holidays('CA', subdiv ='BC')
    df['is_holiday'] = df['Date'].map(lambda x: 1 if x in bc_holidays else 0)
    df['holiday_name'] = df['Date'].map(lambda x: bc_holidays.get(x) if x in bc_holidays else 'normal day')
    rated_dummies = pd.get_dummies(df['holiday_name'], dtype = int)
    df = pd.concat([df, rated_dummies], axis=1)
    return df 

def weekend_feature(df):
    df['is_weekend'] = df['Date'].map(lambda x: 1 if x.weekday() in [5,6] else 0)
    return df 


