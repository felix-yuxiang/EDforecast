import pandas as pd 

# weather_1 = pd.read_csv('climate-daily/climate-daily (1).csv')
# print(weather_1.head)

# Read all the weather data
# weather_df = pd.read_csv('climate-daily/climate-daily (1).csv')
# for i in range(2,26):
#     path = 'climate-daily/climate-daily ('+str(i)+').csv'
#     # print(path)
#     sub_df = pd.read_csv(path)
#     weather_df = weather_df._append(sub_df)

# print(weather_df.shape)

# print(weather_df['LOCAL_DATE'])
# weather_df = weather_df.groupby('LOCAL_DATE')[['MIN_TEMPERATURE','MEAN_TEMPERATURE','MAX_TEMPERATURE',
#     'TOTAL_SNOW','TOTAL_RAIN','TOTAL_PRECIPITATION','HEATING_DEGREE_DAYS','COOLING_DEGREE_DAYS']].mean().reset_index()

# weather_df['LOCAL_DATE'] = pd.to_datetime(weather_df['LOCAL_DATE'])
# print(weather_df.head(20))
# print(weather_df.shape)

# Output weather data (no need to run again anymore)
# weather_df.to_csv('data/weather.csv')

# Output merged weather data
# weather_df.to_csv('data/weather_merged.csv')

def merge_weather(path, n_files, province):
    first_file = path + '/climate-daily (1).csv'
    df = pd.read_csv(first_file)
    for i in range(2,n_files+1):
        p = path + '/climate-daily ('+str(i)+').csv'
        sub_df = pd.read_csv(p)
        df = df._append(sub_df)

    df = df.groupby('LOCAL_DATE')[['MIN_TEMPERATURE','MEAN_TEMPERATURE','MAX_TEMPERATURE',
    'TOTAL_SNOW','TOTAL_RAIN','TOTAL_PRECIPITATION','HEATING_DEGREE_DAYS','COOLING_DEGREE_DAYS']].mean().reset_index()
    df['LOCAL_DATE'] = pd.to_datetime(df['LOCAL_DATE'])
    df['PROVINCE'] = province
    return df

weather_bc = merge_weather('climate-daily/climate-daily-bc', 25, 'BC')
print(weather_bc)
