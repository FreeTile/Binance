import pandas as pd
from binance.client import Client

lines = []
variables = {}
with open('config.txt', 'r') as file:
    lines.extend(file.readlines()[1:19])
with open('config.txt', 'r') as file:
    lines.extend(file.readlines()[8:11])

for line in lines:
    if '=' in line:
        key, value = line.strip().split(' = ')
        variables[key.strip()] = value.strip()

api_key = variables['api_key']
api_secret = variables['api_secret']

client = Client(api_key, api_secret)
bars = client.get_historical_klines(f'{variables["coin1"]}{variables["coin2"]}', eval(f'Client.KLINE_INTERVAL_{variables["clines_time"]}'), variables['date'])

data = pd.DataFrame(bars,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                             'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                             'ignore'])
data = data.drop(['volume', 'close_time', 'quote_asset_volume',
                             'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                             'ignore'], axis=1)
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)

numeric_columns = ['open', 'high', 'low', 'close']
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')
heads=0
legs=0
for i in range(len(data)):
    if data.iloc[i]['open'] > data.iloc[i]['close']:
        heads += data.iloc[i]['high'] - data.iloc[i]['open']
        legs += data.iloc[i]['close'] - data.iloc[i]['low']
    else:
        heads += data.iloc[i]['high'] - data.iloc[i]['close']
        legs += data.iloc[i]['open'] - data.iloc[i]['low']

print('Средняя верхняя тень свечи: ', heads/len(data))
print('Средняя нижняя тень свечи: ', legs/len(data))