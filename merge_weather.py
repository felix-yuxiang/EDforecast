import pandas as pd 

# weather_1 = pd.read_csv('climate-daily/climate-daily (1).csv')
# print(weather_1.head)



# Output merged weather data
# weather_df.to_csv('data/weather_merged.csv')


# Read all the weather data
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
weather_on = merge_weather('climate-daily/climate-daily-on', 19, 'ON')
weather_qc = merge_weather('climate-daily/climate-daily-qc', 25, 'QC')

# Append all dataframes
weather_all = weather_bc._append(weather_on, ignore_index=True)._append(weather_qc, ignore_index=True)
print(weather_all)


# Output merged weather data
weather_all.to_csv('data/weather_merged_all.csv')