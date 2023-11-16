import pandas as pd
import holiday

data = pd.read_csv('data/covid.csv')
weather_data = pd.read_csv('data/weather_merged_all.csv')
# print(data.columns)
# print(weather_data.head())



def sel_province_data(pro_code, pro_name):
    pre_col_name = pro_code+'\n\nNumber of ED visits, pre-pandemic'
    post_col_name = pro_code+'\n\nNumber of ED visits, pandemic period'
    data_pro = data[['Date of visit*',pre_col_name,post_col_name]]
    data_pro['Date of visit*'] = pd.to_datetime(data_pro['Date of visit*'], format='%d-%b-%y')
    # rename the date column and add date for pre-pandemic
    data_pro['Post-pandemic Date'] = data_pro['Date of visit*'].dt.strftime('%Y-%m-%d')
    data_pro['Post-pandemic Date'] = pd.to_datetime(data_pro['Post-pandemic Date'])
    data_pro['Pre-pandemic Date'] = pd.to_datetime('2019-' + data_pro['Post-pandemic Date'].dt.strftime('%m-%d'))

    trimed_data = data_pro[['Pre-pandemic Date',pre_col_name,'Post-pandemic Date',post_col_name]]
    pre_df = trimed_data[['Pre-pandemic Date',pre_col_name]]
    post_df = trimed_data[['Post-pandemic Date',post_col_name]]
    col_name = ['Date','Number_Visits']
    pre_df.columns = col_name
    post_df.columns = col_name
    trimed_data = pd.concat([pre_df,post_df], ignore_index=True)
    trimed_data['Province'] = pro_name

    return trimed_data

trimed_data_bc = sel_province_data('B.C.','BC')
trimed_data_on = sel_province_data('Ont.','ON')
trimed_data_qc = sel_province_data('Que.','QC')

trimed_data_all = trimed_data_bc._append(trimed_data_on,ignore_index=True)._append(trimed_data_qc,ignore_index=True)
print(trimed_data_all)




# date_data = pd.concat([pre_df,post_df])

# # Merge weather data 
weather_data = weather_data.rename(columns={'LOCAL_DATE':'Date','PROVINCE':'Province'})
weather_data['Date'] = pd.to_datetime(weather_data['Date'], format='%Y-%m-%d')
result_data = pd.merge(trimed_data_all, weather_data, on=['Date','Province'], how='inner')
result_data = result_data.drop(columns=['Unnamed: 0']).sort_values(by='Date')
result_data = result_data.reset_index(drop=True)
print(result_data.head())



# # Add holiday feature

# result_data = holiday.holiday_feature(result_data)
# result_data = holiday.weekend_feature(result_data)

# result_data.to_csv('data/output_data.csv')

# Add demographic data
demographic_data = pd.read_csv('data/quarterly_population.csv')[['Year','Quarter','Population at end of quarter']]
demographic_data["Population at end of quarter"] = demographic_data["Population at end of quarter"].str.replace(',', '').astype(int)    
demographic_data["Population at end of quarter(Million)"] = demographic_data["Population at end of quarter"]/1000000
demographic_data = demographic_data.drop(columns=['Population at end of quarter'])
result_data["Quarter"] = result_data['Date'].dt.quarter
result_data["Year"] = result_data['Date'].dt.year
result_data = pd.merge(result_data, demographic_data, on=['Year','Quarter'], how='inner')
result_data = result_data.drop(columns=['Year'])
print(result_data.head())

result_data.to_csv('data/output_data_demographic.csv')

# Eddit ending

# pre_pandemic_weather_data = weather_data.copy()
# post_pandemic_weather_data = weather_data.copy()
# pre_pandemic_weather_data.columns = ['PRE_' + column for column in pre_pandemic_weather_data.columns]
# post_pandemic_weather_data.columns = ['POST_' + column for column in post_pandemic_weather_data.columns]
# pre_pandemic_weather_data['Pre-pandemic Date'] = pre_pandemic_weather_data['PRE_LOCAL_DATE']
# post_pandemic_weather_data['Post-pandemic Date'] = post_pandemic_weather_data['POST_LOCAL_DATE']
# pre_pandemic_weather_data['Pre-pandemic Date'] = pd.to_datetime(pre_pandemic_weather_data['Pre-pandemic Date'])
# post_pandemic_weather_data['Post-pandemic Date'] = pd.to_datetime(post_pandemic_weather_data['Post-pandemic Date'])
# # drop the original date columns
# pre_pandemic_weather_data = pre_pandemic_weather_data.drop(columns=['PRE_LOCAL_DATE'])
# post_pandemic_weather_data = post_pandemic_weather_data.drop(columns=['POST_LOCAL_DATE'])


# result_with_pre = pd.merge(trimed_data, pre_pandemic_weather_data, on='Pre-pandemic Date', how='inner')
# result = pd.merge(result_with_pre, post_pandemic_weather_data, on='Post-pandemic Date', how='inner')
# #dop useless unnamed index column
# result = result.drop(columns=['PRE_Unnamed: 0', 'POST_Unnamed: 0'])
# print(result_with_pre.head())
# print(result.head())

# result.sort_values(by='Post-pandemic Date').to_csv('data/output_data.csv')
