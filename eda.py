import pandas as pd
import holiday
from datetime import datetime, timedelta

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

    trimed_data = trimed_data.drop_duplicates(ignore_index=True)
    return trimed_data

def death_gender_loader(gender):
    df_gender = pd.read_csv(f'data/death_age_{gender}.csv')
    df_gender['Year'] = pd.to_datetime(df_gender['REF_DATE']).dt.year
    df_gender['Week'] = pd.to_datetime(df_gender['REF_DATE'])
    death_data_gender = df_gender[['Year', 'Week', 'GEO', 'VALUE', 'Age at time of death']]
    death_data_gender = death_data_gender.rename(columns={'GEO': 'Province', 'VALUE': 'Death total'})
    death_data_gender[f"Death total {gender}"] = death_data_gender["Death total"].fillna(death_data_gender['Death total'].mean()).astype(int)   
    return death_data_gender

def get_wek_end_date(date):
    input_datetime = date
    first_day_of_week = input_datetime - timedelta(days=input_datetime.weekday())
    saturday_date = first_day_of_week + timedelta(days=5)
    return saturday_date

trimed_data_bc = sel_province_data('B.C.','BC')
trimed_data_on = sel_province_data('Ont.','ON')
trimed_data_qc = sel_province_data('Que.','QC')

trimed_data_bc = holiday.holiday_feature(trimed_data_bc, 'BC')
trimed_data_on = holiday.holiday_feature(trimed_data_on, 'ON')
trimed_data_qc = holiday.holiday_feature(trimed_data_qc, 'QC')


trimed_data_all = trimed_data_bc._append(trimed_data_on,ignore_index=True)._append(trimed_data_qc,ignore_index=True)
# df['holiday_name'] = df['Date'].map(lambda x: bc_holidays.get(x) if x in bc_holidays else 'normal day')
rated_dummies = pd.get_dummies(trimed_data_all['holiday_name'], dtype=int)
trimed_data_all = pd.concat([trimed_data_all, rated_dummies], axis=1)
print(trimed_data_all)
print(trimed_data_all.isnull().sum())




# # date_data = pd.concat([pre_df,post_df])

# # Merge weather data 
weather_data = weather_data.rename(columns={'LOCAL_DATE':'Date','PROVINCE':'Province'})
weather_data['Date'] = pd.to_datetime(weather_data['Date'], format='%Y-%m-%d')
result_data = pd.merge(trimed_data_all, weather_data, on=['Date','Province'], how='inner')
result_data = result_data.drop(columns=['Unnamed: 0']).sort_values(by='Date')
result_data = result_data.reset_index(drop=True)



# # Add holiday feature

# result_data = holiday.holiday_feature(result_data)
result_data = holiday.weekend_feature(result_data)

# result_data[result_data['Province']=='BC'] = holiday.holiday_feature(result_data, 'BC')
# print(result_data[result_data['Date']=='2019-06-24'])
# print(result_data.shape)

result_data.to_csv('data/output_data.csv')

# Add demographic data
df = pd.read_csv('data/demographic_data.csv')
df['Year'] = pd.to_datetime(df['REF_DATE']).dt.year
df['Quarter'] = pd.to_datetime(df['REF_DATE']).dt.quarter
demographic_data = df[['Year', 'Quarter', 'GEO', 'VALUE']]
demographic_data = demographic_data.rename(columns={'GEO': 'Province', 'VALUE': 'Population at end of quarter'})
demographic_data["Population at end of quarter"] = demographic_data["Population at end of quarter"].astype(int)   

# demographic_data["Population at end of quarter(Million)"] = demographic_data["Population at end of quarter"]/1000000
# demographic_data = demographic_data.drop(columns=['Population at end of quarter'])

result_data["Quarter"] = result_data['Date'].dt.quarter
result_data["Year"] = result_data['Date'].dt.year
result_data["Week"] = result_data['Date'].astype(str).map(lambda x: get_wek_end_date(datetime.strptime(x, '%Y-%m-%d')))  

result_data = pd.merge(result_data, demographic_data, on=['Year','Quarter','Province'], how='inner')
gender_list = ['male','female']
gender_data = pd.merge(death_gender_loader('male'), death_gender_loader('female'), on=['Year','Week','Province','Age at time of death'], how='inner')
result_data = pd.merge(result_data, gender_data, on=['Year','Week','Province'], how='inner')
result_data = result_data.drop(columns=['Year', 'Death total_x', 'Death total_y', 'Week'])
result_data.drop_duplicates()
print(result_data.head(20))


# result_data.to_csv('data/output_data_demographic.csv')

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
