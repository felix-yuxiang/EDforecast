import pandas as pd

data = pd.read_csv('data/covid.csv')
weather_data = pd.read_csv('data/weather_merged.csv')
print(data.head())
print(weather_data.head())

# Only keep BC data
data_bc = data[['Date of visit*','B.C.\n\nNumber of ED visits, pre-pandemic','B.C.\n\nNumber of ED visits, pandemic period']]
data_bc['Date of visit*'] = pd.to_datetime(data_bc['Date of visit*'], format='%d-%b-%y')
# rename the date column and add date for pre-pandemic
data_bc['Post-pandemic Date'] = data_bc['Date of visit*'].dt.strftime('%Y-%m-%d')
data_bc['Post-pandemic Date'] = pd.to_datetime(data_bc['Post-pandemic Date'])
data_bc['Pre-pandemic Date'] = pd.to_datetime('2019-' + data_bc['Post-pandemic Date'].dt.strftime('%m-%d'))

trimed_data = data_bc[['Pre-pandemic Date','B.C.\n\nNumber of ED visits, pre-pandemic','Post-pandemic Date','B.C.\n\nNumber of ED visits, pandemic period']]
print(trimed_data.head())

pre_pandemic_weather_data = weather_data.copy()
post_pandemic_weather_data = weather_data.copy()
pre_pandemic_weather_data.columns = ['PRE_' + column for column in pre_pandemic_weather_data.columns]
post_pandemic_weather_data.columns = ['POST_' + column for column in post_pandemic_weather_data.columns]
pre_pandemic_weather_data['Pre-pandemic Date'] = pre_pandemic_weather_data['PRE_LOCAL_DATE']
post_pandemic_weather_data['Post-pandemic Date'] = post_pandemic_weather_data['POST_LOCAL_DATE']
pre_pandemic_weather_data['Pre-pandemic Date'] = pd.to_datetime(pre_pandemic_weather_data['Pre-pandemic Date'])
post_pandemic_weather_data['Post-pandemic Date'] = pd.to_datetime(post_pandemic_weather_data['Post-pandemic Date'])
# drop the original date columns
pre_pandemic_weather_data = pre_pandemic_weather_data.drop(columns=['PRE_LOCAL_DATE'])
post_pandemic_weather_data = post_pandemic_weather_data.drop(columns=['POST_LOCAL_DATE'])


result_with_pre = pd.merge(trimed_data, pre_pandemic_weather_data, on='Pre-pandemic Date', how='inner')
result = pd.merge(result_with_pre, post_pandemic_weather_data, on='Post-pandemic Date', how='inner')
#dop useless unnamed index column
result = result.drop(columns=['PRE_Unnamed: 0', 'POST_Unnamed: 0'])
print(result_with_pre.head())
print(result.head())

result.sort_values(by='Post-pandemic Date').to_csv('data/output_data.csv')